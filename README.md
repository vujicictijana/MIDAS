# Ice-Cream-Dataset

The dataset is generated from the ice-cream factory simulation environmen that is composed of two modules, mixer and pasteurizer.
The values of analog sensors for Mixer level, Pasteurizer level and Pasteurizer temperature are modified using two anomaly injection options: freezing value and step change. The dataset is composed of 115 runs, 40 normal and 75 with anomalies. It is provided as 100 CSV files. 

Columns:
<ul>
  <li>13 parameters for Mixer module</li>
  <li>10 parameters for Pasteurizer module</li>
  <li>Time stamp</li>
  <li>Run id</li>
  <li>Anomaly type</li>
  <li>Sensor where the anomaly was injected</li>
  <li>Actual sensor value</li>
</ul>

Instances:
<ul>
  <li>Total: 799,464</li>
  <li>Normal: 505,046</li>
  <li>Step: 185,109</li>
  <li>Freeze: 109,309</li>
</ul>

