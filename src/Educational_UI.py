#%%
'''This is the educational UI for the heart disease prediction model.
This UI will take in the input parameters from the user and then standardize them with the data provided
in the heart.csv file. The model will then predict the probability of heart disease and provide suggestions
for improving health based on the input parameters.'''
#%%
import numpy as np
import pandas as pd

def standardize(X, y):
    return (y - np.min(X)) / (np.max(X) - np.min(X))

def z_score(X, y):
    return (y - np.mean(X)) / np.std(X)
'''
These are the normalized parameter weights that I can going to use to empahsize
the importance of the features in the dataset. I will do this by taking
the absolute value of the weights and then multiplying them by the normalized input
parameters that users enter into this educational ui application.
[[ 1.63341841] --> age
 [-0.61222105] --> sex
 [-1.67534585] --> cp
 [ 2.49769932] --> trestbps
 [-1.7648317 ] --> chol
 [-1.12889688] --> fbs
 [ 0.39267011] --> restecg
 [ 0.9962049 ] --> thalach
 [ 2.35384067] --> exang
 [-0.78727172] --> oldpeak
 [-2.72367856] --> slope
 [ 1.88789694] --> ca
 [-2.72297551] --> thal
 [-2.34186094]] --> intercept
 (ignore these comments for now since they are just weights for reference,
 but can be used later if any of the major indictors need to be swapped on
 case by case basis)'''

weights = np.array([[-2.72367856], # slope
                    [-2.72297551], # thal
                    [2.49769932], # trestbps
                    [2.35384067], # exang
                    [1.88789694]]) # ca

# Read in the kaggle dataset
data = pd.read_csv('heart.csv')

# Calculate mean and std for each feature in order to gain a reference for unhealthy vital ranges
mean_slope = data['slope'].mean()
std_slope = data['slope'].std()
mean_thal = data['thal'].mean()
std_thal = data['thal'].std()
mean_trestbps = data['trestbps'].mean()
std_trestbps = data['trestbps'].std()
mean_exang = data['exang'].mean()
std_exang = data['exang'].std()
mean_ca = data['ca'].mean()
std_ca = data['ca'].std()
# %%
# ask user for their bioinformation and store them within an array
print('Please enter the following information about yourself:')
temp_slope = float(input("Enter slope (0-2): "))
temp_thal = float(input("Enter thal (0-3): "))
temp_trestbps = float(input("Enter trestbps (90-200): "))
temp_exang = float(input("Enter exang (0-1): "))
temp_ca = float(input("Enter ca (0-3): "))

tslope, tthal, ttrestbps, texang, tca = 0, 0, 0, 0, 0
standardized_list = []

list_of_inputs = [temp_slope, temp_thal, temp_trestbps, temp_exang, temp_ca]
numlist = ['slope', 'thal', 'trestbps', 'exang', 'ca']

for index, value in enumerate(list_of_inputs):
    temp = z_score(data[numlist[index]], value) # standardize the data to find outlier values that have been associated with increase risk of cardiovascular disease
    standardized_list.append(temp)

    if (temp < -1 or temp > 1): # checking if the standardized value is an outlier, and storing the appropriate flag for suggestions later
        if(index == 0):
            print('Your slope is unhealthy')
            tslope = 1
        if(index == 1):
            print('Your thal is unhealthy')
            tthal = 1
        if(index == 2):
            print('Your trestbps is unhealthy')
            ttrestbps = 1
        if(index == 3):
            print('Your exang is unhealthy')
            texang = 1
        if(index == 4):
            print('Your ca is unhealthy')
            tca = 1
t_list = [tslope, tthal, ttrestbps, texang, tca]

'''
The block of code below can be tweaked with greater research to find better lifestyle suggestions for 
improving heart health based on the input parameters that the user provides. Currently, the suggestions are
quite general and may not be specific enough to the user's situation. 

Future Direction: Potentially integrate a more advanced model that can provide personalized suggestions 
based on a wider range of input parameters and medical history. And feed the input parameters into and AI
agent, or specialized LLM, that can provide more tailored advice and recommendations.
'''

print('\nHere are some suggestions to improve your health:\n')
for index, value in enumerate(t_list):
    if value == 1:
        if(index == 0):
            print('''1. Try to reduce your slope by exercising more and getting regular excersice with 50-70 percent of 
your hearts maximum output.\n
2. Try to maintain a generally balanced and healthy diet while avoiding processed foods and excessive sugar and processed fats.\n
3. Try to reduce your overall stress levels and get 8+ hours of sleep per night.\n''')
        if(index == 1):
            print('4. Thalessemia is a group of genetic blood disorders that impact the hemoglobin efficency in your body.\n'
            'please consult a medical professional to seek official help regarding your personal situation.\n')
        if(index == 2):
            print('5. On top of mainting consistent excersice, eating a balanced diet, and managing stress & sleep, make sure to avoid\n'
            'stimulants like caffience or alcohal which can raise your heart rate unnecessarily.\n')
        if(index == 3):
            print('6. Stop exercising immediately if you feel chest pain, breathlessness, or dizziness. Rest and take nitroglycerin ' 
            '\nas prescribed, and seek emergency medical attention if the pain persists.\n')
        if(index == 4):
            print('''7. Heart-Healthy Diet:
Limit saturated and trans fats: These fats contribute to plaque buildup in arteries. 
Choose lean proteins and whole grains: These are good sources of protein and fiber, which are beneficial for heart health. 
Eat plenty of fruits and vegetables: They are rich in antioxidants and fiber, which can help lower your risk of heart disease. 
Limit sodium intake: High sodium can raise blood pressure. \n
8. Increase Physical Activity:
Aim for at least 30 minutes of moderate-intensity exercise most days of the week: This can include brisk walking, jogging, swimming, or cycling. 
Consider strength training: It can help improve cardiovascular health. \n
9. Quit Smoking:
Smoking damages blood vessels and increases the risk of heart disease: Quitting can significantly improve your cardiovascular health. \n
10. Manage Other Risk Factors:
Control high blood pressure and cholesterol: Medications may be necessary to achieve these goals. 
Manage diabetes: High blood sugar levels can damage blood vessels and increase the risk of heart disease. \n
11. Consider Statin Therapy:
Statins are cholesterol-lowering medications: They can help slow down the progression of coronary artery calcification. 
Your doctor will assess your individual risk factors and determine if statin therapy is appropriate for you . ''')
# %%
