import os
import json
import openai
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Draw import rdMolDraw2D
from dotenv import load_dotenv
from pathlib import Path
import sqlite3
from difflib import get_close_matches

print("Exists:", os.path.exists("Q:/lab/Data/chembl_35.db"))

# Load OpenAI API key
load_dotenv(dotenv_path=Path(__file__).parent / "openaikey.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError(" OPENAI_API_KEY not found. Please check your .env file.")

# Known molecule names for fuzzy correction
KNOWN_MOLECULES = [
    "aspirin", "acetaminophen", "ibuprofen", "benzene", "toluene", "ethanol", "methanol",
    "glucose", "fructose", "caffeine", "nicotine", "hydrogen", "oxygen", "carbon",
    "chloroform", "formaldehyde", "water", "sucrose", "citric acid", "acetone"
]

FALLBACK_SMILES = {
    "hydrogen": "[H]", "oxygen": "O", "carbon": "C", "nitrogen": "N"
}

def fuzzy_match_molecule_name(name):
    name = name.lower().strip()
    match = get_close_matches(name, KNOWN_MOLECULES, n=1, cutoff=0.6)
    return match[0] if match else name

# Generate SMILES from molecule name
def get_smiles(molecule_name):
    corrected_name = fuzzy_match_molecule_name(molecule_name)
    print(f" get_smiles(): Requested = {molecule_name}, Corrected = {corrected_name}")  
    if corrected_name in FALLBACK_SMILES:
        return FALLBACK_SMILES[corrected_name]

    prompt = f"Provide ONLY the correct and valid SMILES code for {corrected_name}, without any extra text."
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def interpret_command(text):
    prompt = f"""
You are a chemistry assistant.
Classify the following command:
\"{text}\"

Return a JSON dictionary with one of these intents:
- "generate" (with "molecule": string)
- "modify" (with "actions": list of action objects)
- "navigate" (with "target_node": integer)
- "show_3d"
- "check_existence"
- "fetch_chembl"
- "save"
- "upload"
- "add_history"
- "close_popup"
- "rotate" (with "axis": "x", "y", or "z")
- "stop_spin"
- "jiggle"
- "upload_named" (with "molecule": string)
- "upload_file_choice" (with "index": integer)

 When intent is "modify", use this format ONLY:
[
  {{"action": "remove", "position": 2}},
  {{"action": "replace", "position": 1, "target": "oxygen"}},
  {{"action": "add", "group": "OH", "position": 3}},
  {{"action": "change bond", "position": [1, 2], "bond_type": "double"}}
]

DO NOT return nested fields like 'parameters'. DO NOT rename keys.
Return ONLY valid JSON. No explanation.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        print("[DEBUG] Raw OpenAI response:", raw)
        parsed = json.loads(raw)
        print("Parsed Command:", parsed) 

        # 보정: {"modify": {...}} 형태 → {"intent": "modify", ...}
        if "intent" not in parsed and isinstance(parsed, dict) and len(parsed) == 1:
            intent_key = next(iter(parsed))
            content = parsed[intent_key]
            if isinstance(content, dict):
                parsed = {"intent": intent_key, **content}
            else:
                parsed = {"intent": intent_key}

        # 추가 보정: remove인데 position 없이 quantity만 있는 경우
        if parsed.get("intent") == "modify":
            for action in parsed.get("actions", []):
                if action.get("action") == "remove" and "position" not in action:
                    quantity = action.get("quantity", 1)
                    # 안전하게 뒤쪽 atom index 20~1부터 제거 시도
                    action["position"] = list(range(19, 19 - quantity, -1))  # e.g., [19, 18] if quantity=2

        return parsed

    except Exception as e:
        return {"intent": None, "error": str(e)}



def modify_molecule(mol, commands):
    if mol is None:
        print("Invalid molecule input to modify.")
        return None

    original_smiles = Chem.MolToSmiles(mol)
    mol = Chem.RWMol(mol)

    if isinstance(commands, dict):
        commands = [commands]

    print(f"Starting modification with actions: {commands}")

    try:
        # 🔹 REMOVE
        for command in commands:
            if command.get("action") == "remove":
                pos = command.get("position")
                targets = pos if isinstance(pos, list) else [pos]
                for p in sorted(targets, reverse=True):
                    if isinstance(p, int) and 0 <= p < mol.GetNumAtoms():
                        print(f"Removing atom at position {p}")
                        mol.RemoveAtom(p)
                    else:
                        print(f"Invalid remove position: {p}")

        # 🔹 ADD / REPLACE / BOND
        for command in commands:
            action = command.get("action")
            group = command.get("group")
            pos = command.get("position")
            bond_type = command.get("bond_type")
            target = command.get("target")

            # ADD
            if action == "add" and group:
                try:
                    smiles = get_smiles(group)
                    frag = Chem.MolFromSmiles(smiles)
                    if frag is None:
                        print(f"Invalid fragment for group '{group}': {smiles}")
                        continue
                    num_atoms_before = mol.GetNumAtoms()
                    combined = Chem.CombineMols(mol, frag)
                    mol = Chem.RWMol(combined)
                    new_atom_idx = mol.GetNumAtoms() - 1
                    if isinstance(pos, int) and pos < num_atoms_before:
                        print(f"Adding bond from atom {pos} to new atom {new_atom_idx}")
                        mol.AddBond(pos, new_atom_idx, Chem.BondType.SINGLE)
                except Exception as e:
                    print(f"Failed to add group: {e}")

            # REPLACE
            elif action == "replace" and target is not None:
                if isinstance(pos, int) and 0 <= pos < mol.GetNumAtoms():
                    atom = mol.GetAtomWithIdx(pos)
                    symbol_map = {
                        "hydrogen": "H", "carbon": "C", "nitrogen": "N", "oxygen": "O",
                        "fluorine": "F", "chlorine": "Cl", "bromine": "Br", "iodine": "I"
                    }
                    symbol = symbol_map.get(target.lower(), target)
                    try:
                        atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(symbol)
                        print(f"Replacing atom {pos} with {symbol} ({atomic_num})")
                        atom.SetAtomicNum(atomic_num)
                    except:
                        print(f"Invalid replacement target: {target}")
                else:
                    print(f"Invalid replace position: {pos}")

            # CHANGE BOND
            elif action == "change bond":
                if isinstance(pos, list) and len(pos) == 2:
                    idx1, idx2 = pos
                    bond = mol.GetBondBetweenAtoms(idx1, idx2)
                    if bond:
                        print(f"Changing bond {idx1}-{idx2} to {bond_type}")
                        if bond_type == "single":
                            bond.SetBondType(Chem.BondType.SINGLE)
                        elif bond_type == "double":
                            bond.SetBondType(Chem.BondType.DOUBLE)
                        elif bond_type == "triple":
                            bond.SetBondType(Chem.BondType.TRIPLE)
                        else:
                            print(f"Unknown bond type: {bond_type}")
                    else:
                        print(f"No bond exists between atoms {idx1} and {idx2}")

    except Exception as e:
        print(f"Error during modification actions: {e}")
        return None

    try:
        Chem.SanitizeMol(mol)
        new_smiles = Chem.MolToSmiles(mol)
        print(f"🧬 Original SMILES: {original_smiles}")
        print(f"🧬 Modified SMILES: {new_smiles}")
        if new_smiles == original_smiles:
            print("No effective changes made to molecule.")
            return None
        print(f"Modification result: {new_smiles}")
        modified = Chem.MolFromSmiles(new_smiles)  
        return modified
    except Exception as e:
        print(f" Sanitize error after modification: {e}")
        return None






def get_inchikey(mol):
    return Chem.inchi.MolToInchiKey(mol)

# ✅ Calculate drug-like properties
def compute_molecular_properties(mol):
    try:
        return {
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "H-Bond Donors": Descriptors.NumHDonors(mol),
            "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Refractivity": Descriptors.MolMR(mol),
            "QED": QED.qed(mol),
            "SMILES": Chem.MolToSmiles(mol),
            "InChIKey": Chem.MolToInchiKey(mol)
        }
    except:
        return {}

# ✅ Get ChEMBL ID
def get_chembl_id_from_inchikey(inchikey):
    conn = None
    try:
        db_path = r"Q:\lab\Data\chembl_35.db"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"❌ Database not found at {db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT molregno FROM compound_structures WHERE standard_inchi_key = ?", (inchikey,))
        row = cursor.fetchone()
        if not row:
            return None

        molregno = row[0]
        cursor.execute("SELECT chembl_id FROM molecule_dictionary WHERE molregno = ?", (molregno,))
        chembl_row = cursor.fetchone()

        return chembl_row[0] if chembl_row else None

    except Exception as e:
        print(f"ChEMBL lookup error: {e}")
        return None
    finally:
        if conn:
            conn.close()
