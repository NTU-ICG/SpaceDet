# Class Three: Used to convert tracking result to the form used by WP2
import math
import numpy as np
import pandas as pd
import os

class TrackingDataTransformer:
    def __init__(self, data_folder, camera_name, elevation_deg=15,azimuth_deg=90):
        self.data_folder = data_folder
        self.elevation_deg = elevation_deg
        self.azimuth_deg = azimuth_deg
        self.camera_name = camera_name
      
    @staticmethod    #Calculate transform matrix from body frame to RSW
    def T_Camera_to_RSW(elevation_deg, azimuth_deg):
        elevation_rad = math.radians(elevation_deg)
        azimuth_rad = math.radians(azimuth_deg)
        X_R = np.sin(elevation_rad)
        X_W = np.sin(azimuth_rad)*np.sqrt(1-np.sin(elevation_rad))
        X_S = np.sqrt(1 - X_R**2 - X_W**2)
        Z_axis = [np.sin(elevation_rad),0,np.cos(elevation_rad)]
        Y_axis = [0,-1,0]
        X_axis = [-np.cos(elevation_rad),0,np.sin(elevation_rad)]
        T_Camera_to_RSW = np.column_stack((X_axis, Y_axis, Z_axis))

        return T_Camera_to_RSW

    def Ang1_Ang2(self, A1, A2):    #Angle1 is in the body frame and the definition is from Digantara, Angle2 is under RSW and the definition meets the requirements of the model used by Bai Lu.
        #Translate given bearing angle to body frame
        A1_rad = math.radians(A1)
        A2_rad = math.radians(A2)
        tan_A1 = math.tan(A1_rad)
        tan_A2 = math.tan(A2_rad)
        denominator = math.sqrt(tan_A1**2+tan_A2**2+1)
        z = 1 / denominator
        y = z * tan_A2
        x = z * tan_A1
        body_frame = np.array([x, y, z])

        #Translate from body frame to RSW
        t_camera_to_rsw = self.T_Camera_to_RSW(self.elevation_deg, self.azimuth_deg)
        camera_frame = np.dot(t_camera_to_rsw,body_frame)
        #Calculate bearing angle in RSW with another defination
        A1 = (math.atan(camera_frame[1] / camera_frame[2]) / math.pi) *180
        A2 = (math.atan(camera_frame[0] / camera_frame[2]) / math.pi) *180
        bearing_ang_RSW = [A1,A2]

        return bearing_ang_RSW

    #Conducting transform to data form used by WP2(Now store data into csv, if need can provide np or dataframe directly)
    def executive_transform(self):
        # Read dataset
        file_path = os.path.join('data',self.data_folder,self.camera_name)
        metadata = pd.read_csv(f'{file_path}/Metadata.csv')
        tracking_data = pd.read_csv(f'{file_path}/Tracking_Data_predict.csv')
        output_data = file_path + '/Output_tracking.csv'

        angles_transformed = tracking_data.apply(lambda row: self.Ang1_Ang2(row['Angle1'], row['Angle2']), axis=1)
        # Replace the original angles with the new values
        tracking_data['Angle1'] = [angle[0] for angle in angles_transformed]
        tracking_data['Angle2'] = [angle[1] for angle in angles_transformed]

        # Calculate the transformation matrix for each row and transform the relative position and velocity to RSW
        results = []
        for _, row in metadata.iterrows():
            tracker_position = np.array([row['Px-J2000'], row['Py-J2000'], row['Pz-J2000']])
            tracker_velocity = np.array([row['Vx-J2000'], row['Vy-J2000'], row['Vz-J2000']])

            results.append([row['Timestamp'], row['Px-J2000'], row['Py-J2000'], row['Pz-J2000'], row['Vx-J2000'], row['Vy-J2000'], row['Vz-J2000']])

        # Convert results to a numpy array
        result_array = np.array(results)

        # Step 1: Convert result_array to a DataFrame
        columns_result = ['Timestamp', 'Tracker_Px', 'Tracker_Py', 'Tracker_Pz', 'Tracker_Vx', 'Tracker_Vy', 'Tracker_Vz']
        result_df = pd.DataFrame(result_array, columns=columns_result)
        tracking_data['Object'] = tracking_data['Object'].astype(int)

        # Step 2: Merge the two dataframes based on Timestamp and Object
        merged_data = pd.merge(tracking_data, result_df, on=['Timestamp'])

        # Step 3: Select and reorder the columns
        final_columns = [
            'Object', 'Timestamp', 'Angle1', 'Angle2',
            'Tracker_Px', 'Tracker_Py', 'Tracker_Pz', 
            'Tracker_Vx', 'Tracker_Vy', 'Tracker_Vz'
        ]
        final_data = merged_data[final_columns]

        # Step 4: Sort the DataFrame based on Timestamp
        final_data = final_data.sort_values(by=['Object', 'Timestamp'])

        final_data.to_csv(output_data, index=False)
        # Save or further process the final_data DataFrame as required

# if __name__ == "__main__":
#     elevation_deg=15
#     azimuth_deg=90
#     data_folder = 'D:/space_dataset/Datasets/parameters/New_Version/Cam_2-Az_90'
#     transformer = TrackingDataTransformer(data_folder,elevation_deg,azimuth_deg)
#     transformer.executive_transform()
#     print("Success!")
        
        
        


