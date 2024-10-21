from rdkit import Chem
from rdkit.Chem import AllChem
import polars
import os
import uuid
from typing import List
from openbabel import pybel

from pdbfixer import PDBFixer
from openmm.app import PDBFile

from vina import Vina


#
# Ligand Prep Functions
#
def writeLigandsToFileFromDF(df, ligandsPath) -> None:
    """
    Converts PDBBlocks to PDBQT and writes them to a Ligands directory
    to a .PDBQT format#

    Args:
        df (polars Dataframe) : Dataframe containing a column with PDBBlock information
        ligandsPath (str) : string of the path to the '/Ligands' directory.

    Returns:
        N/A

    """

    # For now, just assuming that the Ligands directory is in same directory as the script
    PATH = os.getcwd() + "/" + ligandsPath

    # Iterate over each row. This should have guard rails put up, I know
    for row in df.rows(named=True):
        fname = PATH + "/" + row["ID"] + ".pdbqt"
        PDBBlock = row["PDBBlock"]
        pybelMOL = pybel.readstring("PDB", PDBBlock)
        pybelMOL.write("PDBQT", fname)


def embedMolecules(mol):
    """
    Takes a mol representation of a molecule and embeds it (converts it to 3d rep)

    Args:
        mol (rdkit mol) : rdkit mol representaiton

    Returns
        mol (rdkit mol) : 3d version of mol

    """
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol)

    return mol


def generatePDBBlockFromMol(mol) -> str:
    """
    Converts a PDB format to a string format

    Args:
        mol (rdkit mol) : mol represenation of a mol

    Returns
        PDBBlock (str) : string representation of a PDB Block
    """
    return Chem.MolToPDBBlock(mol, flavor=4)


def addPDBBlockToDF(df):
    """
    Adds embedded 3D RDKit mols to a column of a DF

    Args:
        df (polars Dataframe) : Dataframe containing a column with 3dRDKIT mol information

    Returns:
        df (polars Dataframe) : Dataframe containing a column with added PDBBlock string column



    """

    # I'm pretty sure the map_elements function isn't optimal here, but it should be fine here
    df = df.with_columns(
        polars.col("3DrdkitMOL")
        .map_elements(lambda x: generatePDBBlockFromMol(x))
        .alias("PDBBlock")
    )
    return df


def addRDKit3DmolsToDF(df):
    """
    Adds embedded 3D RDKit mols to a column of a DF

    Args:
        df (polars Dataframe) : Dataframe containing a column with rdkitMOL column

    Returns:
        df (polars Dataframe) : Dataframe containing a column with 3dRDKIT mol information



    """
    df = df.with_columns(
        polars.col("rdkitMOL")
        .map_elements(lambda x: embedMolecules(x))
        .alias("3DrdkitMOL")
    )
    return df


def generateRDKitMolsFromDF(df):
    """
    Generates RDKIT mols from SMILES and attaches to a polars dataframe

    Args:
        df (polars dataframe) : Dataframe with a 'SMILES' columns

    Returns
        df (polars dataframe) : Dataframe with an added 'rdkitMOL' column
    """
    validMols = []

    for row in df.rows(named=True):
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is not None:
            validMols.append(mol)
        else:
            validMols.append(None)

    # Add mol representations
    df = df.with_columns(polars.Series(name="rdkitMOL", values=validMols))
    # Drop rows that aren't valid mols
    df = df.drop_nulls()
    return df


def readCsv(path: str, separator: str = ",", hasHeaders: bool = False):
    """
    Reads a CSV file with a SMILES column.
    Args:
        path (str) : path to the file
    Returns:
        dataframe (dataframe) : polars dataframe of the relevant data
    """
    if os.path.isfile(path):
        return polars.read_csv(
            path,
            separator=separator,
            has_header=hasHeaders,
            infer_schema_length=10000000,
        )
    else:
        return FileNotFoundError


def generateMolID() -> str:
    """
    Generates a ten digit alphanumerica id for a compound, prefixed with 'DO-' (eight digits random, prefixed with 'DO-').
    Has coverage of ~1.2e12 for minimial chances of namne clash.
    Args:
        None
    Returns:
        idstring (str) : ten digit alphanumeric string.
    """

    # Is this necessary? No. But I'm two cocktails deep right now.
    PREFIX = "DO-"
    while True:
        random_id = str(uuid.uuid4())
        random_id = random_id.upper()
        random_id = random_id.replace("-", "")
        idstring = random_id[0:8]
        first = idstring[0]
        last = idstring[-1]
        if not first.isdigit() and not last.isdigit():
            break
    return PREFIX + idstring


def getSmiles(
    df,
    headerName: List[str] = ["Smiles", "SMILES", "smiles"],
    *args: str,
):
    """
    Attempts to find a column in a Dataframe that matches a Smiles column (e.g. 'Smiles', 'SMILES', 'smiles') if not provided,
    and returns list of SMILES
    Args:
        df (polars Dataframe) : Dataframe containing N columns, one of which should be labeled as 'Smiles' | 'SMILES' | 'smiles'
        *args (str) : optional *args holder for compoundID identifier
    Returns:
        outDF (polars DataFrame) : Dataframe containing SMILES and ID column
    """
    # instantiate
    outDF = polars.DataFrame()
    # Weakly check overlap
    existingColumns = df.columns
    # Check if compoundID column exists; if it doesn't, generate it
    if len(args) > 0:
        compoundIDs = list(df.get_column(args))
    else:
        compoundIDs = [generateMolID() for _ in range(len(df))]

    # Check if the headerName for SMILES exists
    if len(headerName) > 1:
        overlap = list(set(existingColumns) & set(headerName))
        SMILES = list(df.column(overlap[0]))
    else:
        SMILES = list(df.get_column(headerName[0]))
    # Add columns to the datframe
    outDF = outDF.with_columns(polars.Series(name="SMILES", values=SMILES))
    outDF = outDF.with_columns(polars.Series(name="ID", values=compoundIDs))
    return outDF


# PROTEIN PREP FUNCTIONS
def fixPDB(path: str, structureName: str):
    """
    'Fixes' a PDB structure by adding missing residues, finding/adding missing residues,
    and writes out corrected files.

    Args:
        path (str) : path to the PDB structure to be fixed
        structureName (str) : name of the PDB structure (e.g. 4HW2)

    Returns:

        N/A
    """

    fname = path + "/" + structureName + ".pdb"
    fixer = PDBFixer(filename=fname)
    fixer.findMissingResidues()
    # fixer.findNonstandardResidues()
    # fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(
        fixer.topology, fixer.positions, open(f"Protein/{structureName}_fixed.pdb", "w")
    )


def convertPDBtoPDBQT(path: str, structureName: str):
    """
    Converts a PDB to PDBQT for docking

    Args:
        path (str) : path to the Protein directory
        structureName (str) : base name of the protein not including path or extension (e.g. 4HW2)
    Returns:
        N/A
    """
    fname = path + "/" + structureName + "_fixed.pdb"
    ofname = path + "/" + structureName + "_fixed.pdbqt"
    pdb = next(pybel.readfile("PDB", fname))
    pdb.write("pdbqt", ofname, overwrite=True, opt={"r": None})


# Vina Execution Code
def createVinaInstance(receptorPath: str):
    """
    Creates a base vina instance that will run all of the docking calculations

    Args:
        receptorPath (str) : path to the Receptor that will be used for docking (PDBQT format)
    Returns:
        vinaEngine (vina class obj) : vina engine that runs calculations


    """
    # These are fixed coordinates for now,
    # but there will be a process to get these dynamically later

    center_x = 36.92
    center_y = -8.67
    center_z = 24.385

    # Instantiate the vina backend
    vinaEngine = Vina(sf_name="vina")
    vinaEngine.set_receptor(receptorPath)
    vinaEngine.compute_vina_maps(
        center=[center_x, center_y, center_z], box_size=[20, 20, 20]
    )

    return vinaEngine


def addDockingScoresToDF(scores: List, df):
    """
    Adds the final docking scores to the Dataframe

    Args:
        scores (list(dictionary)) : list of dictionaries that are organized  {'ID' : id, 'dockingScore' : score}
        df (polars dataframe) : polars dataframe that contains all the input information
    Returns:
        df (polars dataframe) : polars dataframe that contains all the input information
    """

    scores = polars.DataFrame(scores)

    df = df.join(scores, on="ID")

    return df


def dockLigands(df, proteinDir: str, ligandDir: str, outDir: str, vinaEngine):
    """
    Performs the actual docking, using Autodock Vina. It will simply look in the "Ligands"
    and "Protein" files and dock all of them, using the configure.txt file (WHICH MUST ALREADY EXIST!).
    It will edit the --ligand and --ouput parameters for each compound on-the-fly

    Args:
        df (polars df) : polars dataframe containing all of the compounds (ids and embedded ligand) that will be docked
        proteinDir (str) : path to the protein that will be used for the docking job
        ligandDir (str) : path to the ligands directory
        outDir (str) : path of where the docked poses will be put for output
        vinaEngine (vina obj) : vina instance that will run the calculations
    Returns:


    """

    # Scores dict to hold all of the pose energies
    scores = []

    for row in df.rows(named=True):
        poseOut = outDir + row["ID"] + "_docked.pdbqt"
        ligand = ligandDir + "/" + row["ID"] + ".pdbqt"
        vinaEngine.set_ligand_from_file(ligand)
        vinaEngine.dock(exhaustiveness=16, n_poses=5)
        vinaEngine.write_poses(poseOut, n_poses=5, overwrite=True)
        temp = {}
        temp["ID"] = row["ID"]
        temp["dockingScore"] = vinaEngine.score()[0]
        scores.append(temp)

    return addDockingScoresToDF(scores, df)


# For now assuming the Ligands folder is in the same directory as the script
ligandsFilepath = "Ligands"
proteinFilepath = "Protein"
structureName = "4HW2"
preppedStructurePath = proteinFilepath + "/" + structureName + "_fixed.pdbqt"
outDir = "dockOutput/"

smilesInFilepath = "input/SMALLTEST.csv"
separator = ","
compoundIdHeader = "Molecule ChEMBL ID"
smilesHeader = ["Smiles"]

# Vars needed to eventually chose the right binary of Vina to use (GPU/CPU, Linux/OSX/Window$)
# OS = "LINUX"
# COMPUTE = "CPU"


# Ligand Routine
# Yes this breaks a lot of rules and i dont care
df = readCsv(smilesInFilepath, separator, True)
outDF = getSmiles(df, smilesHeader)


outDF = generateRDKitMolsFromDF(outDF)
outDF = addRDKit3DmolsToDF(outDF)
outDF = addPDBBlockToDF(outDF)
writeLigandsToFileFromDF(outDF, ligandsFilepath)


# Protein Routine for fixing structure
fixPDB(proteinFilepath, structureName)
convertPDBtoPDBQT(proteinFilepath, structureName)

# Instantiate the vina engine
vinaEngine = createVinaInstance(preppedStructurePath)


# Run the docking
outDF = dockLigands(outDF, proteinFilepath, ligandsFilepath, outDir, vinaEngine)
outDF = outDF.select(["SMILES", "ID", "dockingScore"])
# Just save the dataframe for now...
outDF.write_csv("dockingOutputSummary.csv")
