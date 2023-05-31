import json
import datetime

results = {}
with open('output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/inference_model_final_wo_optim/ycbv_test/results.json', 'r') as f:
    data = json.load(f)
    for key in data:
        if not key.startswith("0/"):
            results[key] = {
                "obj_id": data[key][0]["obj_id"],
                "bbos_est": data[key][0]["bbox_est"],
                "R": data[key][0]["R"],
                "t": data[key][0]["t"]
            }

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"my_results/results_{timestamp}.json"
with open(filename, 'w') as f:
    json.dump(results, f, indent=4)

