# MIDAS: Modular Ice-cream factory Dataset on Anomalies in Sensors

<h2>Dataset</h2>

<p align="justify"> The dataset is generated from the ice-cream factory simulation environmen that is composed of six modules (Mixer, Pasteurizer, Homogenizer, Aeging Cooling, Dynamic Freezer, and Hardening). The values of analog sensors for level and temperature are modified using three anomaly injection options: freezing value, step change and ramp change. The dataset is composed of 1000 runs, out of which 258 were executed without anomalies. 
 
Files: 
<ul>
  <li>1000 CSV files, one file for each run</li>
  <li>file name contains the run id and type (Normal, Freeze, Ramp, or Step)</li>
</ul>

Dataset division:
<ul>
  <li>Training: runs from 1 to 500 </li>
  <li>Validation: runs from 501 to 600 </li>
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


<h2>Experiments</h2>

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

<h5> Citation: </h5>

Please cite the following publication when referring to MIDAS: Markovic, T., Leon, M., Leander, B., & Punnekkat, S. (2023). <a href="https://ieeexplore.ieee.org/document/10058956"> A Modular Ice Cream Factory Dataset on Anomalies in Sensors to Support Machine Learning Research in Manufacturing Systems</a>. IEEE Access.



<h5> Acknowledgment </h5>
This work has been partially supported by the H2020 ECSEL EU Project <a href="www.insectt.eu"> Intelligent Secure Trustable Things (InSecTT)</a>
and  ECSEL Joint Undertaking (JU) project <a href="https://dais-project.eu/">  Distributed Artificial Intelligent System (DAIS)</a> . It reflects only the author’s view and the Commission is not responsible for any use that may be made of the information it contains.



