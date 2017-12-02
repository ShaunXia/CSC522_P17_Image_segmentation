import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

image_recall, image_precision = [0.709677], [0.651692]
true_recall = [0.640945, 0.652916, 0.65392, 0.543373, 0.917117, 0.765358]
true_precision = [0.994999, 0.984415, 0.996813, 0.953636, 0.951319, 0.938048]
plt.figure(1)
plt.plot(image_recall, image_precision, "x")
plt.plot(true_recall, true_precision, "ro")
plt.axis([0, 1, 0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
red_patch = mpatches.Patch(color='red', label='Ground Truth')
blue_patch = mpatches.Patch(color='blue', label='Result')
plt.legend(handles=[red_patch, blue_patch])

plt.show()
