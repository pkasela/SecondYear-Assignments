import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;  
import java.util.Random;

public class WekaMLP {

	public static void main(String[] args) throws Exception {
	    CSVLoader loader = new CSVLoader();
	    loader.setSource(new File("/home/pranav/Desktop/AdvancedML-Assignments/AML/Assignment5/data.csv"));
	    Instances InitialData = loader.getDataSet();
	    InitialData.setClassIndex(InitialData.numAttributes() - 1);
	    
	    NumericToNominal convert= new NumericToNominal();
	    String[] options= new String[2];
        options[0]="-R";
        options[1]="last";
        convert.setOptions(options);
	    convert.setInputFormat(InitialData);
	    
	    Instances data =  Filter.useFilter(InitialData, convert);
	    
	    Evaluation evaluation = new Evaluation(data);
	    Instances train = data.trainCV(5, 0);
	    Instances test  = data.testCV(5, 0);
	    
	    MultilayerPerceptron mlp = new MultilayerPerceptron();
	    /* The following options are defined as:
	     * -L learning rate
	     * -M momentum
	     * -N number of iterations
	     * -V validation size
	     * -S seed
	     * -E max error in training
	     * -H number of neurons in each parameters
	     * -G if you want GUI
	     */
	    mlp.setOptions(Utils.splitOptions("-L 0.0836 -M 0.1587 -N 200 -V 0 -S 0 -E 20 -H 4,2"));
	    //mlp.setOptions(Utils.splitOptions("-G")); //usa la GUI
	    mlp.buildClassifier(train);
	    evaluation.evaluateModel(mlp, test);
	    System.out.println(evaluation.toMatrixString());
	}

}
