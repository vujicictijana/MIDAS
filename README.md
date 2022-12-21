# MIDAS: Modular Ice-cream factory Dataset on Anomalies in Sensors

<p align="justify"> The dataset is generated from the ice-cream factory simulation environmen that is composed of six modules (Mixer, Pasteurizer, Homogenizer, Aeging Cooling, Dynamic Freezer, and Hardening). The values of analog sensors for level and temperature are modified using three anomaly injection options: freezing value, step change and ramp change. The dataset is composed of 1000 runs, out of which 258 were executed without anomalies. 
 
Files: 
<ul>
  <li>1000 CSV files, one file for each run</li>
  <li>file name contains the run id and type (Normal, Freeze, Ramp, or Step)</li>
</ul>

Dataset division:
<ul>
  <li>Training: runs from 1 to 600 </li>
  <li>Testing: runs from 601 to 1000 </li>
</ul>

Columns:
<ul>
  <li>ordinal number of instance within one run</li>
  <li>13 parameters for Mixer module</li>
  <li>8 parameters for Pasteurizer module</li>
  <li>4 parameters for Homogenizer module</li>
  <li>7 parameters for AgeingCooling module</li>
  <li>16 parameter's for DynamicFreezing module</li>
  <li>6 parameters for Hardening module</li>
  <li>Time stamp</li>
  <li>Anomaly type</li>
  <li>Sensor where the anomaly was injected</li>
  <li>Actual sensor value</li>
</ul>

Classes:
<ul>
  <li>Normal </li>
  <li>Freeze</li>
  <li>Step</li>
  <li>Ramp</li>
</ul>

Instances:
<ul>
  <li>Total: 36,124,859 </li>
  <li>Normal: 17,422,215 (49.67%)</li>
  <li>Anomalies: 18,182,644 (50.33%) </li>
</ul>


Repository contains code for experiments for two different problems Anomaly Detection (AD) and Anomaly Classification (AC).

Files for experiments:
<ul>
 <li>1_get_data.ipynb - reads CSV files and transforms them into dataframes with 500-100-400 runs for training, validation and testing, respectively</li>
 <li>2_DT.ipynb - Training Decision Tree and saving the model </li>
 <li>3_RF.ipynb - Training Random Forest with different numbers of DTs (varying from 5 to 50 with increment of 5) and saving the best model </li>
 <li>4_LR.ipynb - Training Logistic Regression and saving the model </li>
 <li>3_RF.ipynb - Training MultiLayer Perceptron with different numbers of neurons in the hidden layer (varying from 5 to 50 with increment of 5) and saving the best model</li>
 <li>6_test.ipynb - Testing saved models </li>

</ul>



