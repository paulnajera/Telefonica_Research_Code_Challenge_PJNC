

"""
Define number of clients, batch size per client
num_clients: between how many people, groups or whatever (clients) the data is going to be distributed
num_rounds:  interaction between server and clients, number of times the model goes to the client to update
             weights and model
batch_size_per_client: it represents the number of samples used in one forward and backward pass through the network
             and has a direct impact on the accuracy and computational efficiency of the training process.


num_classes_per_client: in non_iid_data_split, the data distribution across clients is not identical. it could, 
             for example, group similar classes of data together or assign different subsets of data to different 
             clients. Therefore, the number of classes is the number of subsets or conglomeration of classes that are
             assigned to each client

"""

num_clients = 50
batch_size_per_client = 32
num_rounds = 10
num_attributes = 2
attributes = ['Male', 'Young', 'Black_Hair']

"""
Possible options for attributes:
5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair 
Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones 
Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline 
Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace 
Wearing_Necktie Young
"""


"""

0                1               2          3               4    5     6        7        8          9          10     11         12             13      14         15           16   17        18            19             20   21                   22       23          24      25        26        27          28                29          30         31      32           33         34              35          36               37               38              39                                                                                                                                                                  
5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young

"""

"""
1.Crear el modelo
2.Entrenar el modelo
3.Evaluar el modelo
4.Tunning the mode
5.Testear el modelo


Tipo de de
"""

