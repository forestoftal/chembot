import wx
from PIL import Image as PILImage
import io
import os
import json
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import chembot
import webbrowser
from whisper_wrapper import transcribe_audio 
import sqlite3
from pathlib import Path
from difflib import get_close_matches
from command_correction import correct_command, normalize_spoken_file_name
SAVE_DIR = Path.home() / "chembot"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

rdMolDraw2D = Draw.rdMolDraw2D

def normalize_spoken_file_name(raw_text):
    import re
    words = raw_text.lower().split()
    name_parts = []
    for word in words:
        if word.isdigit():
            name_parts.append(word)
        elif re.match(r'^[a-zA-Z]+$', word):
            name_parts.append(word)
    return "_".join(name_parts)

class TreeNode:
    def __init__(self, smiles, image, number=None, parent=None, inchikey=None, chembl_id=None):
        self.smiles = smiles
        self.image = image
        self.number = number
        self.parent = parent
        self.children = []
        self.x = 0
        self.y = 0
        self.inchikey = inchikey
        self.chembl_id = chembl_id

    def add_child(self, child_node):
        self.children.append(child_node)

class TreePanel(wx.ScrolledWindow):
    def __init__(self, parent):
        super().__init__(parent, style=wx.VSCROLL | wx.HSCROLL)
        self.root_node = None
        self.SetScrollRate(20, 20)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnClick)
        self.node_positions = {}
        self.click_callback = None
        self.nodes_by_number = {}
        self.image_rects = {}
        self.image_size = (120, 120)

    def set_root(self, root_node):
        self.root_node = root_node
        self.calculate_positions()
        self.Refresh()

    def normalize_spoken_file_name(raw_text):
        import re
        # e.g., "open aspirin 1 2" → "aspirin_1_2"
        words = raw_text.lower().split()
        name_parts = []
        for word in words:
            if word.isdigit():
                name_parts.append(word)
            elif re.match(r'^[a-zA-Z]+$', word):
                name_parts.append(word)
        filename = "_".join(name_parts)
        return filename


    def calculate_positions(self):
        def assign_positions(node, depth=0, x_offset=0):
            if not node.children:
                node.x = x_offset
                node.y = depth * (self.image_size[1] + 120)
                self.node_positions[node.number] = (node.x, node.y)
                return x_offset + self.image_size[0] + 60

            child_x = x_offset
            child_centers = []
            for child in node.children:
                child_x = assign_positions(child, depth + 1, child_x)
                child_centers.append(self.node_positions[child.number][0])

            min_x = min(child_centers)
            max_x = max(child_centers)
            node.x = (min_x + max_x) // 2
            node.y = depth * (self.image_size[1] + 120)
            self.node_positions[node.number] = (node.x, node.y)
            return child_x

        if self.root_node:
            self.node_positions.clear()
            total_nodes = self.count_nodes(self.root_node)
            self.adjust_image_size(total_nodes)

            assign_positions(self.root_node)

            all_x = [pos[0] for pos in self.node_positions.values()]
            all_y = [pos[1] for pos in self.node_positions.values()]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            visible_width = self.GetSize().GetWidth()
            panel_padding = 100
            extra_top_margin = 80

            horizontal_shift = max(0, (visible_width - (max_x - min_x)) // 2 - min_x + panel_padding)

            for key in self.node_positions:
                x, y = self.node_positions[key]
                self.node_positions[key] = (x + horizontal_shift, y + extra_top_margin)

            final_x = [x for x, _ in self.node_positions.values()]
            final_y = [y for _, y in self.node_positions.values()]
            self.SetVirtualSize((max(final_x) + 200, max(final_y) + 200))

    def count_nodes(self, node):
        return 1 + sum(self.count_nodes(child) for child in node.children)

    def adjust_image_size(self, total_nodes):
        if total_nodes > 30:
            self.image_size = (70, 70)
        elif total_nodes > 15:
            self.image_size = (90, 90)
        else:
            self.image_size = (120, 120)

    def save_unique_file(self, base_path):
        """
        Given a base Path object like '~/chembot/aspirin_1.json',
        return a new Path that doesn't overwrite existing files.
        E.g., aspirin_1_2.json, aspirin_1_3.json, etc.
        """
        if not base_path.exists():
            return base_path

        base_name = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        counter = 2
        while True:
            new_path = parent / f"{base_name}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1
     


    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()
        if not self.root_node:
            return
        self.image_rects.clear()

        def draw_connections(node):
            for child in node.children:
                x1, y1 = self.node_positions[node.number]
                x2, y2 = self.node_positions[child.number]
                dc.DrawLine(x1, y1 + self.image_size[1] // 2 + 10, x2, y2 - self.image_size[1] // 2 - 10)
                draw_connections(child)

        def draw_nodes(node):
            x, y = self.node_positions[node.number]
            if node.image:
                bmp = wx.Bitmap(node.image.Scale(self.image_size[0], self.image_size[1]))
                img_w, img_h = bmp.GetSize()
                img_x = x - img_w // 2
                img_y = y - img_h // 2
                dc.DrawBitmap(bmp, img_x, img_y, True)
                self.image_rects[node.number] = wx.Rect(img_x, img_y, img_w, img_h)

            label = f"{node.number}"
            if node.chembl_id:
                label += f" ({node.chembl_id})"
            tw, th = dc.GetTextExtent(label)
            dc.DrawText(label, x - tw // 2, y + self.image_size[1] // 2 + 10)
            for child in node.children:
                draw_nodes(child)

        draw_connections(self.root_node)
        draw_nodes(self.root_node)

    def OnClick(self, event):
        pos = self.CalcUnscrolledPosition(event.GetPosition())
        for node_num, rect in self.image_rects.items():
            if rect.Contains(pos):
                if self.click_callback:
                    self.click_callback(node_num)
                break

    def list_saved_files(self):
        return sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".json")])

    def load_file_by_name(self, filename):
        path = SAVE_DIR / filename
        if not path.exists():
            wx.MessageBox(f"File '{filename}' not found.", "Error")
            return
        try:
            with open(path, 'r') as file:
                data = json.load(file)
                root_data = data.get("root")
                if root_data:
                    self.click_callback.load_tree_from_data(root_data)
                    wx.MessageBox(f"Loaded {filename}", "Upload Success")
                else:
                    wx.MessageBox("Root node not found in file.", "Error")
        except Exception as e:
            wx.MessageBox(f"Failed to load file: {e}", "Error")

def get_chembl_id_from_inchikey(inchikey):
    try:
        db_path = r"Q:\lab\Data\chembl_35.db"
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
        print(f"ChEMBL ID fetch error: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

    
class ChemBotGUI(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Chem-Bot: AI Molecule Editor", size=(1800, 900))
        panel = wx.Panel(self)
        
        outer_sizer = wx.BoxSizer(wx.VERTICAL)

        # Add a text box to show voice input feedback and transcription
        self.output_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(600, 80))
        outer_sizer.Add(self.output_text, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)

        control_panel = wx.Panel(panel)
        control_sizer = wx.BoxSizer(wx.HORIZONTAL)
        control_panel.SetSizer(control_sizer)

        self.input_text = wx.TextCtrl(control_panel, size=(300, -1))
        self.submit_btn = wx.Button(control_panel, label="Submit")
        self.submit_btn.Bind(wx.EVT_BUTTON, self.on_submit_text)

        self.speak_btn = wx.Button(control_panel, label="Speak")
        self.speak_btn.Bind(wx.EVT_BUTTON, self.on_speak_command)

        self.view3d_btn = wx.Button(control_panel, label="3D Structure")
        self.view3d_btn.Bind(wx.EVT_BUTTON, self.on_view_3d)

        self.check_btn = wx.Button(control_panel, label="Check Existence")
        self.check_btn.Bind(wx.EVT_BUTTON, self.on_check_existence)

        self.save_file_btn = wx.Button(control_panel, label="Save Molecule")
        self.save_file_btn.Bind(wx.EVT_BUTTON, self.on_save_file)

        self.chembl_info_btn = wx.Button(control_panel, label="ChEMBL Info")
        self.chembl_info_btn.Bind(wx.EVT_BUTTON, self.on_fetch_chembl_info)
        control_sizer.Add(self.chembl_info_btn, flag=wx.RIGHT, border=10)

        self.upload_btn = wx.Button(control_panel, label="Show File")
        self.upload_btn.Bind(wx.EVT_BUTTON, self.on_upload_file)
        control_sizer.Add(self.upload_btn, flag=wx.RIGHT, border=10)

        self.active_dialog = None
        self.active_browser_path = None


        control_sizer.Add(wx.StaticText(control_panel, label="Command:"), flag=wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border=5)
        control_sizer.Add(self.input_text, flag=wx.RIGHT, border=10)
        control_sizer.Add(self.submit_btn, flag=wx.RIGHT, border=10)
        control_sizer.Add(self.speak_btn, flag=wx.RIGHT, border=10)
        control_sizer.Add(self.view3d_btn, flag=wx.RIGHT, border=10)
        control_sizer.Add(self.check_btn, flag=wx.RIGHT, border=10)
        control_sizer.Add(self.save_file_btn, flag=wx.RIGHT, border=10)
        

        outer_sizer.Add(control_panel, flag=wx.EXPAND | wx.ALL, border=10)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.mol_bitmap = wx.StaticBitmap(panel, size=(500, 500))
        self.add_to_tree_btn = wx.Button(panel, label="Add to History")
        self.add_to_tree_btn.Bind(wx.EVT_BUTTON, self.on_add_to_history)
        left_sizer.Add(self.add_to_tree_btn, flag=wx.ALL | wx.ALIGN_CENTER, border=5)

        left_sizer.Add(self.mol_bitmap, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        self.prop_display = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(500, 200))
        left_sizer.Add(self.prop_display, flag=wx.LEFT | wx.RIGHT | wx.BOTTOM, border=10)
        
        logo_path = "/home/ssu/chembl_db/chembl_logo_small.png"  # Your actual logo path
        if os.path.exists(logo_path):
            logo_image = wx.Image(logo_path, wx.BITMAP_TYPE_PNG)
            logo_image.Rescale(150, 60)  # smaller width x height
            logo_bitmap = wx.StaticBitmap(panel, bitmap=wx.Bitmap(logo_image))
            left_sizer.Add(logo_bitmap, flag=wx.LEFT | wx.BOTTOM, border=10)

        self.tree_panel = TreePanel(panel)
        self.tree_panel.SetMinSize((1300, 800))
        self.tree_panel.click_callback = self



        main_sizer.Add(left_sizer)
        main_sizer.Add(self.tree_panel, flag=wx.EXPAND | wx.ALL, border=10)

        outer_sizer.Add(main_sizer, proportion=1, flag=wx.EXPAND)
        panel.SetSizer(outer_sizer)

        self.root_node = None
        self.current_node = None
        self.node_counter = 1
        self.nodes_by_number = {}

        self.Centre()
        self.Show()
        panel.Layout()

    def process_input(self, text):
        self.input_text.SetValue(text)  # Show transcription in input line
        self.handle_user_command(text)  # This should already exist to process commands


    def on_submit_text(self, event):
        self.handle_natural_input(self.input_text.GetValue())

    def on_speak_command(self, event):
        self.output_text.SetValue("Voice input activated...\n Recording...")
        self.input_text.SetValue("Recording...")  # Shows in input box too
        try:
            transcription = self.transcribe_and_correct()
            self.output_text.SetValue(f"📝 Transcription result:\n{transcription}")
            self.input_text.SetValue(transcription)
            self.process_input(transcription)
        except Exception as e:
            self.output_text.SetValue("Error in voice input")
            self.input_text.SetValue("Error in voice input")
            print("Error in voice input:", e)


    def transcribe_and_correct(self):
        import sounddevice as sd
        import soundfile as sf
        from faster_whisper import WhisperModel

        fs = 44100
        duration = 5
        filename = "mic_input.wav"

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        sf.write(filename, recording, fs)

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(filename, language="en")

        text = ""
        for segment in segments:
            text += segment.text.strip() + " "
        return self.correct_command(text.strip())


    def correct_command(self, raw_text):
        dictionary = {
            "draw benzene": "draw benzene",
            "benzene ring": "draw benzene",
            "3d": "3D structure",
            "three d": "3D structure",
            "add oh": "add OH",
            "replace": "replace group",
            "remove": "remove group",
            "upload": "upload file",
            "generate": "generate molecule"
        }

        raw_text = raw_text.lower()
        best_match = get_close_matches(raw_text, dictionary.keys(), n=1, cutoff=0.7)
        return dictionary[best_match[0]] if best_match else raw_text
    
    def handle_user_command(self, text):
        self.handle_natural_input(text)


    def handle_natural_input(self, raw_text):
        try:
            #SHOW FILE
            if "show file" in raw_text.lower():
                files = self.tree_panel.list_saved_files()
                if not files:
                    self.prop_display.SetValue(" No saved files found in ~/chembot.")
                else:
                    file_list = "\n".join([f"{i+1}. {f}" for i, f in enumerate(files)])
                    self.prop_display.SetValue(f" Saved Files in ~/chembot:\n\n{file_list}")
                return

            # UPLOAD BY INDEX or FUZZY NAME
            if any(kw in raw_text.lower() for kw in ["upload", "open", "load"]):
                files = self.tree_panel.list_saved_files()
                raw = raw_text.lower()

                # By order: "upload second one"
                order_words = {
                    "first": 0, "1st": 0, "one": 0,
                    "second": 1, "2nd": 1, "two": 1,
                    "third": 2, "3rd": 2, "three": 2,
                    "fourth": 3, "4th": 3, "four": 3,
                    "fifth": 4, "5th": 4, "five": 4
                }
                for word in raw.split():
                    if word in order_words:
                        idx = order_words[word]
                        if idx < len(files):
                            self.tree_panel.load_file_by_name(files[idx])
                            return
                        else:
                            wx.MessageBox("That file number does not exist.", "Error")
                            return

                # Try exact filename (if mentioned)
                for file in files:
                    name_only = file.lower().replace(".json", "")
                    if name_only in raw.replace(" ", "_") or name_only in raw:
                        self.tree_panel.load_file_by_name(file)
                        return

                # Fallback fuzzy match
                from difflib import get_close_matches
                filename_guess = normalize_spoken_file_name(raw_text)
                matches = get_close_matches(filename_guess + ".json", files, n=1, cutoff=0.6)
                if matches:
                    self.tree_panel.load_file_by_name(matches[0])
                    return

                wx.MessageBox("No matching file found to upload.", "Error")
                return

            # Otherwise: parse structured commands
            parsed = chembot.interpret_command(raw_text)
            if not parsed or not isinstance(parsed, dict):
                wx.MessageBox("Command interpretation failed: no response from OpenAI.", "Error")
                return
            intent = parsed.get("intent")
            if not intent:
                wx.MessageBox(f"No intent detected. OpenAI response: {parsed}", "Unknown Command")
                return

            if intent == "generate":
                smiles = chembot.get_smiles(parsed.get("molecule"))
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    raise ValueError("Could not parse SMILES.")
                self.molecule_name = parsed.get("molecule", "molecule")
                img = self.draw_numbered_molecule(mol, size=(120, 120))
                inchikey = chembot.get_inchikey(mol)
                chembl_id = chembot.get_chembl_id_from_inchikey(inchikey)
                self.pending_node = TreeNode(smiles, img, number=None, inchikey=inchikey, chembl_id=chembl_id)
                self.display_molecule(mol)
                wx.MessageBox("Molecule generated. Click 'Add to History' to keep it.", "Pending Molecule")

            elif intent == "modify":
                if not self.current_node:
                    wx.MessageBox("No current molecule selected to modify.", "Error")
                    return

                actions = parsed.get("actions")
                if not isinstance(actions, list):
                    wx.MessageBox("Invalid modification actions. Please try again.", "Error")
                    return

                mol = Chem.MolFromSmiles(self.current_node.smiles)
                modified = chembot.modify_molecule(mol, actions)

                if modified:
                    self.node_counter += 1
                    img = self.draw_numbered_molecule(modified, size=(120, 120))
                    inchikey = chembot.get_inchikey(modified)
                    chembl_id = chembot.get_chembl_id_from_inchikey(inchikey)
                    self.pending_node = TreeNode(
                        Chem.MolToSmiles(modified), img, self.node_counter,
                        inchikey=inchikey, chembl_id=chembl_id, parent=self.current_node
                    )
                    self.display_molecule(modified)
                    wx.MessageBox("Molecule modified. Click ' Add to History' to keep it.", "Pending Molecule")
                else:
                    wx.MessageBox("Modification failed. Please check your command.", "Error")




            elif intent == "show_3d":
                self.on_view_3d()

            elif intent == "check_existence":
                self.on_check_existence(None)

            elif intent == "fetch_chembl":
                self.on_fetch_chembl_info(None)

            elif intent == "save":
                self.on_save_file(None)

            elif intent == "upload":
                self.on_upload_file(None)

            elif intent == "add_history":
                self.on_add_to_history(None)

            elif intent == "navigate":
                target = parsed.get("target_node")
                if isinstance(target, int) and target in self.nodes_by_number:
                    self.on_click_history_node(target)
                else:
                    wx.MessageBox(f"Node {target} not found.", "Error")

            elif intent == "close_popup":
                self.close_popup()

            elif intent == "rotate":
                axis = parsed.get("axis")
                self.on_view_3d(axis=axis)

            elif intent == "stop_spin":
                self.on_view_3d(stop_spin=True)

            elif intent == "jiggle":
                self.on_view_3d(jiggle=True)

            else:
                wx.MessageBox(f"Unknown command intent: {intent}", "Warning")

        except Exception as e:
            wx.MessageBox(f"Command Error: {e}", "Error", wx.ICON_ERROR)


    def draw_numbered_molecule(self, mol, size=(300, 300)):
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        for i, atom in enumerate(mol.GetAtoms()):
            opts.atomLabels[atom.GetIdx()] = str(i)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return wx.Image(io.BytesIO(drawer.GetDrawingText()), wx.BITMAP_TYPE_PNG)

    def display_molecule(self, mol):
        img = self.draw_numbered_molecule(mol, size=(500, 500))
        self.mol_bitmap.SetBitmap(wx.Bitmap(img))
        props = chembot.compute_molecular_properties(mol)
        if props:
            self.prop_display.SetValue("\n".join(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in props.items()))

    def on_click_history_node(self, node_number):
        node = self.nodes_by_number.get(node_number)
        if node:
            mol = Chem.MolFromSmiles(node.smiles)
            if mol:
                self.current_node = node
                self.display_molecule(mol)
                self.tree_panel.set_root(self.root_node)

    def on_view_3d(self, event=None, axis=None, stop_spin=False, jiggle=False):
        if not self.current_node or not self.current_node.smiles:
            wx.MessageBox("No molecule selected for 3D view!", "Error")
            return
        try:
            mol = Chem.MolFromSmiles(self.current_node.smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            mol_block = Chem.MolToMolBlock(mol)
            spin_script = ""
            if axis == "x":
                spin_script = "viewer.spin('x');"
            elif axis == "y":
                spin_script = "viewer.spin('y');"
            elif axis == "z":
                spin_script = "viewer.spin('z');"
            elif stop_spin:
                spin_script = "viewer.stopSpin();"
            elif jiggle:
                spin_script = """
                    let angle = 0;
                    setInterval(() => {
                        angle += 5;
                        viewer.rotate(angle % 360, 'y');
                        viewer.render();
                    }, 100);
                """
            elif axis == "y":
                spin_script = "viewer.spin('y');"
            html = f"""
            <html><body>
            <script src='https://3Dmol.org/build/3Dmol-min.js'></script>
            <div style='width:100%; height:100%; position: absolute;' id='viewer'></div>
            <script>
              let viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
              viewer.addModel(`{mol_block}`, "mol");
              viewer.setStyle({{}}, {{stick:{{}}}});
              viewer.zoomTo();
              {spin_script}
              viewer.render();
            </script>
            </body></html>
            """
            path = os.path.abspath("viewer.html")
            with open(path, "w") as f:
                f.write(html)
            self.active_browser_path = path
            webbrowser.open(f"file://{path}")

        except Exception as e:
            wx.MessageBox(f"3D View Error: {e}", "Error")

    def on_check_existence(self, event):
        inchikey = getattr(self.current_node, 'inchikey', None)
        if not inchikey:
            wx.MessageBox("No InChIKey available.", "Error")
            return
        try:
            with open(r"Q:\lab\Data\inchi_keys.txt", 'r') as f:
                found = any(inchikey == line.strip() for line in f)
            msg = "Exists in ChEMBL!" if found else "🚀 New compound — patent it!"
            wx.MessageBox(msg, "Check Result")
        except Exception as e:
            wx.MessageBox(f"Error reading inchi_keys.txt: {e}", "Error")

    def on_save_file(self, event=None):
        if not self.current_node:
            wx.MessageBox("No molecule selected to save.", "Error")
            return

        def clone_upward_path(node):
            new_node = TreeNode(
                smiles=node.smiles,
                image=node.image,
                number=node.number,
                parent=None,
                inchikey=node.inchikey,
                chembl_id=node.chembl_id
            )
            if node.children:
                counter = [node.number + 1]  # start new counter
                new_node.children = [clone_subtree(c, counter) for c in node.children]
            if node.parent:
                parent = clone_upward_path(node.parent)
                parent.add_child(new_node)
                return parent
            else:
                return new_node

        def clone_subtree(node, counter):
            new_node = TreeNode(
                smiles=node.smiles,
                image=node.image,
                number=counter[0],
                inchikey=node.inchikey,
                chembl_id=node.chembl_id
            )
            counter[0] += 1
            new_node.children = [clone_subtree(c, counter) for c in node.children]
            return new_node

        def serialize(node):
            return {
                "smiles": node.smiles,
                "number": node.number,
                "inchikey": node.inchikey,
                "chembl_id": node.chembl_id,
                "children": [serialize(child) for child in node.children]
            }

        # 🔹 Determine safe molecule name
        smiles = self.current_node.smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol_name = getattr(self, "molecule_name", "molecule")
        except:
            mol_name = "molecule"

        # 🔹 Prepare filename
        safe_name = mol_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        node_num = self.current_node.number
        filename_base = f"{safe_name}_node_{node_num}"

        save_dir = Path.home() / "chembot"
        save_dir.mkdir(parents=True, exist_ok=True)

        i = 1
        while (save_dir / f"{filename_base}_{i}.json").exists():
            i += 1
        filename = f"{filename_base}_{i}.json"
        save_path = save_dir / filename

        # 🔹 Clone and save
        subtree_root = clone_upward_path(self.current_node)
        try:
            with open(save_path, 'w') as f:
                print("DEBUG SERIALIZATION:", serialize(subtree_root))
                json.dump({"root": serialize(subtree_root)}, f, indent=2)
            wx.MessageBox(f"Saved as {save_path}", "Saved", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Error saving file: {e}", "Error")

    def on_upload_file(self, event):
        with wx.FileDialog(self, "Open Molecule File", wildcard="JSON files (*.json)|*.json",
                        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            path = fileDialog.GetPath()

            # SET molecule name from filename
            filename = Path(path).name
            if "_node_" in filename:
                self.molecule_name = filename.split("_node_")[0]
            else:
                self.molecule_name = "molecule"

            try:
                with open(path, 'r') as file:
                    data = json.load(file)
                    root_data = data.get("root")
                    if root_data:
                        self.load_tree_from_data(root_data)
                    else:
                        wx.MessageBox("파일에서 루트 노드를 찾을 수 없습니다.", "에러")
            except Exception as e:
                wx.MessageBox(f"파일 로드 실패: {e}", "에러")


    def load_json_file(self, path):
        try:
            with open(path, 'r') as file:
                data = json.load(file)
                root_data = data.get("root")
                if root_data:
                    self.load_tree_from_data(root_data)
                    wx.MessageBox(f"Loaded {os.path.basename(path)}", "Upload Success")
                else:
                    wx.MessageBox("Root node not found in file.", "Error")
        except Exception as e:
            wx.MessageBox(f"Failed to load file: {e}", "Error")



    def load_tree_from_data(self, data, parent=None):
        mol = Chem.MolFromSmiles(data["smiles"])
        img = self.draw_numbered_molecule(mol, size=(120, 120)) if mol else None
        node = TreeNode(data["smiles"], img, data["number"], parent, inchikey=data.get("inchikey"))
        if parent:
            parent.add_child(node)
        else:
            self.root_node = node

        self.nodes_by_number[data["number"]] = node
        self.current_node = node  # 가장 마지막 노드를 current로 설정

        for child_data in data.get("children", []):
            self.load_tree_from_data(child_data, node)

        self.display_molecule(mol)
        self.tree_panel.set_root(self.root_node)
        self.reindex_tree()

    def on_fetch_chembl_info(self, event):
        inchikey = getattr(self.current_node, 'inchikey', None)
        if not inchikey:
            wx.MessageBox("No InChIKey found. Generate a molecule first.", "Error")
            return

        try:
            import sqlite3
            db_path = r"Q:\lab\Data\chembl_35.db" 

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Step 1: Get molregno from InChIKey
            cursor.execute("SELECT molregno FROM compound_structures WHERE standard_inchi_key = ?", (inchikey,))
            row = cursor.fetchone()
            if not row:
                wx.MessageBox("No matching entry in ChEMBL.", "Info")
                return
            molregno = row[0]

            # Step 2: Define categories and table mappings
            table_groups = {
                "🧬 Molecule Info": ["molecule_dictionary", "compound_structures", "compound_properties"],
                "🧾 Synonyms & Hierarchy": ["molecule_synonyms", "molecule_hierarchy"],
                "🧪 Bioactivities": ["activities"],
                "💊 Drug Indications": ["drug_indication"],
                "⚙️ Mechanism of Action": ["drug_mechanism"],
                "🚨 Structural Alerts": ["compound_structural_alerts"],
                "💉 Formulations & Warnings": ["formulations", "drug_warning"],
                "🧫 Biotherapeutics": ["biotherapeutics", "biotherapeutic_components"],
                "🧩 Classification": ["molecule_atc_classification", "molecule_frac_classification", "molecule_hrac_classification", "molecule_irac_classification"],
                "📦 Compound Records": ["compound_records"]
            }

            grouped_info = {}

            for group_name, tables in table_groups.items():
                group_lines = []
                for table in tables:
                    try:
                        cursor.execute(f"SELECT * FROM {table} WHERE molregno = ?", (molregno,))
                        rows = cursor.fetchall()
                        if rows:
                            col_names = [desc[0] for desc in cursor.description]
                            for r in rows:
                                line = ", ".join(f"{col}: {val}" for col, val in zip(col_names, r))
                                group_lines.append(f"• {line}")
                    except Exception as e:
                        group_lines.append(f"Skipped table {table}: {e}")
                if group_lines:
                    grouped_info[group_name] = group_lines

            conn.close()

            if grouped_info:
                display_text = ""
                for section, lines in grouped_info.items():
                    display_text += f"{section}\n" + "-"*len(section) + "\n"
                    display_text += "\n".join(lines) + "\n\n"

                self.active_dialog = wx.Dialog(self, title="ChEMBL Molecule Info", size=(1000, 600))
                dlg = self.active_dialog  # keep compatibility
                box = wx.BoxSizer(wx.VERTICAL)
                text = wx.TextCtrl(dlg, value=display_text, style=wx.TE_MULTILINE | wx.TE_READONLY)
                box.Add(text, 1, wx.EXPAND | wx.ALL, 10)
                dlg.SetSizer(box)
                dlg.ShowModal()
                self.active_dialog.Destroy()
                self.active_dialog = None

            else:
                wx.MessageBox("No data found in linked tables.", "Info")

        except Exception as e:
            wx.MessageBox(f"Error querying ChEMBL: {e}", "Error")

    def reindex_tree(self):
        """트리 내 노드에 대해 1부터 번호 다시 부여"""
        counter = [1]  # 리스트로 해야 내부 함수에서 값 변경 가능

        def assign_number(node):
            node.number = counter[0]
            self.nodes_by_number[node.number] = node
            counter[0] += 1
            for child in node.children:
                assign_number(child)

        self.nodes_by_number.clear()
        if self.root_node:
            assign_number(self.root_node)

    def on_add_to_history(self, event=None):
        if not hasattr(self, 'pending_node'):
            wx.MessageBox("No molecule pending to add. Please generate or modify first.", "Info")
            return

        node = self.pending_node
        parent = self.current_node

        if parent:
            parent.add_child(node)
        else:
            self.root_node = node

        self.reindex_tree()
        self.current_node = node
        self.tree_panel.set_root(self.root_node)
        del self.pending_node
        wx.MessageBox("Molecule added to history!", "Info")





    def get_next_node_number(self):
        numbers = [node.number for node in self.tree_nodes if node.number is not None]
        return max(numbers) + 1 if numbers else 1
    
    def close_popup(self):
        if self.active_dialog:
            self.active_dialog.Destroy()
            self.active_dialog = None
            wx.MessageBox("Popup closed.", "Info")
        elif self.active_browser_path:
            os.remove(self.active_browser_path)  # Optional: remove the HTML file
            self.active_browser_path = None
            wx.MessageBox("3D viewer closed (please manually close the browser tab).", "Info")
        else:
            wx.MessageBox("No popup is currently open.", "Info")


    def display_file_list(self, files):
        if not files:
            self.prop_display.SetValue("No saved files found in ~/chembot.")
            return
        file_list = "\n".join(f"• {f}" for f in files)
        self.prop_display.SetValue(f"Files in ~/chembot:\n\n{file_list}")
        bmp = wx.Bitmap(500, 500)
        dc = wx.MemoryDC(bmp)
        dc.Clear()
        dc.SelectObject(wx.NullBitmap)
        self.mol_bitmap.SetBitmap(bmp)  # Blank image area





if __name__ == "__main__":
    print("Starting ChemBot GUI...")
    app = wx.App(False)
    frame = ChemBotGUI()
    app.MainLoop()
