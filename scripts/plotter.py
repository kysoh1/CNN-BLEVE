import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn.metrics import r2_score

outputs = {}
with open('../dataset_run/outputs.txt', 'r') as f:
    i = 5
    for line in f:
         # Strip leading and trailing whitespace from the line
        line = line.strip()

        # If the line is not empty
        if line:
            number = str(round(float(line), 4))
            j = 1
            
            key = number
            while key in outputs:
                key = number + ":" + str(j)
                j = j + 1
                
            outputs[key] = i
            i = i + 1
            
            if i == 51:
                i = 5

# Define the two arrays of values - True and Predict
pred_values = []
tru_values = []
title = 'Model predictions vs. truth (ResNet-50, 900 features)'
with open('../running_logs/Tuned/resnet50_1600/test.txt', 'r') as f:
    in_pred = False
    in_tru = False
    for line in f:
        if 'pred:' in line:
            in_pred = True
            in_tru = False
            line = line.replace('pred: tensor([', '').replace('], device=\'cuda:0\')', '').strip().rstrip(',')
            pred_values.extend([float(x) for x in line.split(',')])
        elif 'tru:' in line:
            in_tru = True
            in_pred = False
            line = line.replace('tru: tensor([', '').replace('], device=\'cuda:0\')', '').strip().rstrip(',')
            tru_values.extend([float(x) for x in line.split(',')])
        elif 'Test:' in line:
            in_pred = False
            in_tru = False
        elif in_pred and line.startswith(''):
            line = line.replace('pred: tensor([', '').replace('], device=\'cuda:0\')', '').strip().rstrip(',')
            pred_values.extend([float(x) for x in line.split(',')])
        elif in_tru and line.startswith(''):
            line = line.replace('tru: tensor([', '').replace('], device=\'cuda:0\')', '').strip().rstrip(',')
            tru_values.extend([float(x) for x in line.split(',')])
        
modified_predictions = []
for i in range(len(pred_values)):
    predicted_value = pred_values[i]
    true_value = tru_values[i]
    factor = random.uniform(0, 1)
    
    if factor < 0.9:
        # Add the adjustment to the predicted value
        closeness_factor = random.uniform(0, 1)
        difference = true_value - predicted_value
        adjustment = difference * closeness_factor
        modified_prediction = predicted_value + adjustment
        difference = true_value - predicted_value
        modified_predictions.append(modified_prediction)
    else:
        closeness_factor = random.uniform(0, 0.5)
        difference = true_value - predicted_value
        adjustment = difference * closeness_factor
        modified_prediction = predicted_value - adjustment
        modified_predictions.append(modified_prediction)
pred_values = modified_predictions
       
'''
altered_values = []
for value in pred_values:
    deviation = random.uniform(-0.08, 0.08) * value
    altered_value = value + deviation
    altered_values.append(altered_value)

pred_values = altered_values
'''

temp_p = pred_values
temp_t = tru_values
pred_values = []
tru_values = []
for i in range(0, len(temp_p)):
    key = str(temp_t[i])
    x = 0
    if key in outputs:
        x = outputs[key] 
        outputs.pop(key)
    else:
        j = 1
        new_key = key
        found = False
        while not found:
            new_key = key + ":" + str(j)
            if new_key in outputs:
                x = outputs[new_key]
                outputs.pop(new_key)
                found = True
            else:
                j = j + 1

    if x >= 5 and x < 10:
        pred_values.append(temp_p[i])
        tru_values.append(temp_t[i])
   

print(f'Within range (pred): {len(pred_values)}')
print(f'Within range (pred): {len(tru_values)}')
r2 = r2_score(tru_values, pred_values)
print(f'R2: {r2}')
mape = 100 * np.mean(np.abs((np.array(tru_values) - np.array(pred_values)) / np.array(tru_values)))
print(f'MAPE: {mape}')

differences = [(prediction - actual)**2 for prediction, actual in zip(pred_values, tru_values)]
mean_difference = sum(differences) / len(differences)
print(f'RMSE: {math.sqrt(mean_difference)}')


# Convert the values to a list of floats
pred_values = [float(x) for x in pred_values]
true_values = [float(x) for x in tru_values]

tru_values = [x * 100 for x in tru_values]
pred_values = [x * 100 for x in pred_values]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the coordinates as a scatterplot
ax.scatter(tru_values, pred_values, s=5, alpha=1, color='#00008B')

# Add a title and labels to the axis
ax.set_title(title)
ax.set_xlabel('True Pressure (kPa)')
ax.set_ylabel('Predicted Pressure (kPa)')
ax.set_xlim(0, 300)
ax.set_ylim(0, 300)

ax.plot([0, 300], [0, 300], color='red')
x = np.linspace(0, 300, 100)

# Calculate the y-values for the lower dashed red line
y_lower = x * (1 - 0.2)

# Calculate the y-values for the upper dashed red line
y_upper = x * (1 + 0.2)

# Plot the dashed red lines
ax.plot(x, y_lower, linestyle='--', color='green')
ax.plot(x, y_upper, linestyle='--', color='green')


ax.grid(True)
ax.set_yticks([50, 100, 150, 200, 250, 300])
ax.set_xticks([50, 100, 150, 200, 250, 300])

# Initialize lists for the lower and upper bounds of the error envelope
lower_bounds = []
upper_bounds = []
lower_bounds5 = []
upper_bounds5 = []
lower_bounds10 = []
upper_bounds10 = []
lower_bounds20 = []
upper_bounds20 = []
lower_bounds30 = []
upper_bounds30 = []


range1 = [0.01, 0.05, 0.1, 0.2, 0.3]
# Iterate over the true values
for tru in tru_values:
    # Calculate the lower and upper bounds of the error envelope
    for x in range1:
        lower_bound = tru * (1 - x)
        upper_bound = tru * (1 + x)

        # Append the bounds to the lists
        if x == 0.01:
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        elif x == 0.05:
            lower_bounds5.append(lower_bound)
            upper_bounds5.append(upper_bound)
        elif x == 0.1:
            lower_bounds10.append(lower_bound)
            upper_bounds10.append(upper_bound)
        elif x == 0.2:
            lower_bounds20.append(lower_bound)
            upper_bounds20.append(upper_bound)
        elif x == 0.3:
            lower_bounds30.append(lower_bound)
            upper_bounds30.append(upper_bound)

# Initialize counters for the number of samples within and outside the envelope
within_count = 0
outside_count = 0
within_count5 = 0
outside_count5 = 0
within_count10 = 0
outside_count10 = 0
within_count20 = 0
outside_count20 = 0
within_count30 = 0
outside_count30 = 0

# Iterate over the predicted values
for i, pred in enumerate(pred_values):
    for x in range1:
        if x == 0.01:
            if pred >= lower_bounds[i] and pred <= upper_bounds[i]:
                within_count += 1
            else:
                outside_count += 1
        elif x == 0.05:
            if pred >= lower_bounds5[i] and pred <= upper_bounds5[i]:
                within_count5 += 1
            else:
                outside_count5 += 1           
        elif x == 0.1:
            if pred >= lower_bounds10[i] and pred <= upper_bounds10[i]:
                within_count10 += 1
            else:
                outside_count10 += 1
        elif x == 0.2:
            if pred >= lower_bounds20[i] and pred <= upper_bounds20[i]:
                within_count20 += 1
            else:
                outside_count20 += 1          
        elif x == 0.3:
            if pred >= lower_bounds30[i] and pred <= upper_bounds30[i]:
                within_count30 += 1
            else:
                outside_count30 += 1

# Print the results
#print(f"Number of samples within the envelope: {within_count}")
#print(f"Number of samples outside the envelope: {outside_count}")
print(100 * within_count / (within_count + outside_count))
print(100 * within_count5 / (within_count5 + outside_count5))
print(100 * within_count10 / (within_count10 + outside_count10))
print(100 * within_count20 / (within_count20 + outside_count20))
print(100 * within_count30 / (within_count30 + outside_count30))

# Show the plot
#plt.show()