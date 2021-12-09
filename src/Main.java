import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;

public class Main {
    public static void main(String args[]) throws Exception {
        DataSource source = new DataSource("D:\\Prog\\Java\\WEKA\\resources\\house.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);
        LinearRegression model = new LinearRegression();
        model.buildClassifier(dataset);
        System.out.println("LR FORMULA : " + model);
        Instance myHouse = dataset.lastInstance();
        double price = model.classifyInstance(myHouse);
        System.out.println("-------------------------");
        System.out.println("PREDICTING THE PRICE : " + price);
    }
}
