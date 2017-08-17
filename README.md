# CarEvaluation
Use machine learning to evaluate cars

Hi,everybody.The project is done while I was learning machine learning.  
This uses basic machine learning algorithm, k-Nearest Neighbor(KNN)  
Let's take a look at the program flow:  
(1)Collecting data ---> function: filecarmatrix(filepath)   
   The dataset used is car.txt that is illustrated in car描述.txt  
(2)Prepare data --->function: autoNorm(DataSet)    
   This function is used for normalizing data    
(3)Classifier --->function: classifyCar(CarData, DataSet, Labels, k)    
(4)Test Classifier ---> function: CarEvaClassTest()     
   Use 1/10 of the data as test data to test the function.   
(5)Interactive   ---> function: CarEvaluation()     
   The function is intended to increase interactivity.    
    
 Example description,you can see the specific program notes
