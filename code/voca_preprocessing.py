import os

original_data_path = '../data/databases/voca'
output_data_path = '../data/voca'
for i in range(8, 9):
    for folder_id in os.listdir(os.path.join(original_data_path, f'imagessubject{i}')):
        if not folder_id.startswith('FaceTalk'):
            continue
        for sentence_id in os.listdir(os.path.join(original_data_path, f'imagessubject{i}', folder_id)):
            cur_path = os.path.join(original_data_path, f'imagessubject{i}', folder_id, sentence_id)
            cur_out = []
            for frame_id in os.listdir(cur_path):
                if '26_C' in frame_id:
                    cur_out.append(frame_id)
            cur_out.sort() 
            # copy all frames to output_data_path, maining structure
            cur_out_path_structure = os.path.join(output_data_path, f'imagessubject{i}', folder_id, sentence_id)
            if not os.path.exists(cur_out_path_structure):
                os.makedirs(cur_out_path_structure)
            for frame_id in cur_out:
                os.system(f'cp {os.path.join(cur_path, frame_id)} {os.path.join(cur_out_path_structure)}')