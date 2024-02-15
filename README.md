# Data snipper home task
Fine tune a DistilBert model on the Finer-139 data set


## Trained models
You can find two trained models on this page.
https://huggingface.co/gvisser


## converting to ONNX
I have created a Google Colab sheet for exporting a model to ONNX format
https://colab.research.google.com/drive/12GLnjEDythUKSwVJipzKr_5gz4nHpriI?usp=sharing<br>

You can finde the models on google drive<br>
https://drive.google.com/drive/folders/1p8Du-qhEfmLZnArEGvH5Dg1DcmEAMDni<br>
The best model is `Datasnipper_FinerDistilBert.onnx`


## C# application
A console application that asks for a path to an ONNX model and a piece of text. It will proceed to do inference with the provided with the model on the text
Use the shortcut in the project root to use the application.

If the text space is left blank the following is used as default:
```
"For both the three and six months ended May 31, 2016, 312 homesites were sold to Lennar by one of the Company's unconsolidated entities for $ 92.0 million that resulted in $29.7 million, of gross profit, of which the Company's portion was deferred. For the three months ended May 31, 2015, Lennar Homebuilding equity in earnings included $ 11.6 million of equity in earnings from one of the Company's unconsolidated entities primarily due to sales of approximately 60 homesites and a commercial property to third parties for $ 121.3 million that resulted in $ 37.6 million of gross profit. For the six months ended May 31, 2015, Lennar Homebuilding equity in earnings included $ 43.0 million of equity in earnings from one of the Company's unconsolidated entities primarily due to (1) sales of approximately 660 homesites to third parties for $ 407.2 million that resulted in $ 138.4 million of gross profit and (2) sales of 300 homesites to Lennar for $ 126.4 million that resulted in $ 44.6 million of gross profit, of which the Company's portion was deferred. Balance Sheets On May 2, 2016 (the \"Closing Date” ), the Company contributed, or obtained the right to contribute, its investment in three strategic joint ventures previously managed by Five Point Communities in exchange for an investment in a newly formed Five Point entity."
```
This is part of the Finer-139 data set. Two Named Entities should be recognized as part of the data set:
1.  earnings included $ 11.6 million of equity -> B-IncomeLossFromEquityMethodInvestments
2.  earnings included $ 43.0 million of equity -> B-IncomeLossFromEquityMethodInvestments


### Note.
You'll find a VSTO addin folder. Due to dependency issues and time I chose to go in a different direction.