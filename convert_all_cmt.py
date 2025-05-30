import os
from cmt_utils import parse_cmt_xml_to_instance, save_instance_for_runpy_format, validate_cmt_pkl_format

CMT_FOLDER = "C:\\Users\\Martin\\Documents\\Studium\\Angewandtes Wissenschaftliches Arbeiten\\VRP-GYM\\datasets\\christofides-et-al-1979-cmt"
OUTPUT_FOLDER = "C:\\Users\\Martin\\Documents\\Studium\\Angewandtes Wissenschaftliches Arbeiten\\attention-learn-to-route\\data\\cmt"
CMT_FILES = ["CMT01.xml", "CMT02.xml", "CMT03.xml", "CMT04.xml", "CMT05.xml", "CMT11.xml", "CMT12.xml"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in CMT_FILES:
    xml_path = os.path.join(CMT_FOLDER, filename)
    base = os.path.splitext(filename)[0].lower()

    pkl_path = os.path.join(OUTPUT_FOLDER, f"{base}_sdvrp.pkl")
    json_path = os.path.join(OUTPUT_FOLDER, f"{base}_meta.json")

    print(f"📄 Verarbeite: {filename}")
    instance, meta = parse_cmt_xml_to_instance(xml_path)
    save_instance_for_runpy_format(instance, meta, pkl_path, json_path)

print("\n🧪 Starte Validierung:")
validate_cmt_pkl_format(OUTPUT_FOLDER)