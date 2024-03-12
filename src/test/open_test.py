import torch
import numpy as np

 #open-test -> image : open, joint : open
def test_predict_joint_open(model, image_data : np, joint_data : np, joint_image_path : str, data_name : str, random_index_array : list):

    image_data_narray = image_data.copy()
    joint_data_narray = joint_data.copy()

    data_num_in_1image = 1
    #joint_predictions = torch.zeros((n, 159, 14))
    inputs_image = np.zeros((data_num_in_1image, 80, 3, 32, 32))
    inputs_joint = np.zeros((data_num_in_1image, 80, joint_dim))
    for i in range(data_num_in_1image):
        input_image = image_data_narray[random_index_array[i]]
        joint_datum = joint_data_narray[random_index_array[i]]

        inputs_image[i] = input_image 
        inputs_joint[i] = joint_datum

    joint_input_data = inputs_joint[:, 0:-1, :]
    joint_target_data = inputs_joint[:, 1 :, :] 
    image_inputs_data = inputs_image[:, 0:-1, :, :, :]
    
    #create_goal_image_data_function is (np->np)_function
    image_goal = create_goal_image_data(inputs_image[:, 1:, :, :, :])
    
    joint_input_data = torch.from_numpy(joint_input_data).float()
    joint_target_data = torch.from_numpy(joint_target_data).float()
    image_inputs_data = torch.from_numpy(image_inputs_data).float()
    image_goal = torch.from_numpy(image_goal).float()

    joint_predictions, _, _ = model.forward(image_inputs_data.reshape(-1, 3, 32, 32), joint_input_data, image_goal.reshape(-1, 3, 32, 32))
    
    visualize_joint_test(joint_predictions, joint_target_data, data_num_in_1image, joint_image_path + "/open_" + data_name)