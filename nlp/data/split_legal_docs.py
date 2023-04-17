import os
import json

from msc.utils.utility import get_root_path

def main():
    data_dir = os.path.join(get_root_path(), 'data')
    shr_json = os.path.join(data_dir, 'legal_documents', 'shr_dataset.json')
    with open(shr_json, 'r') as f:
        shr = json.load(f)
        for key in shr:
            case_json = {"text": shr[key]["text"], "summary": shr[key]["summary"]}
            # save case_json to file
            if not case_json["summary"] == "" and not case_json["text"] == "":
                with open(os.path.join(data_dir, 'json', key + '.json'), 'w') as f:
                    json.dump(case_json, f)
        
if __name__ == '__main__':
    main()
    