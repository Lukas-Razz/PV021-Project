package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.optimizers.SGD;
import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import cz.pv021.neuralnets.network.MultilayerPerceptron;
import cz.pv021.neuralnets.utils.Pair;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cz.pv021.neuralnets.optimizers.Optimizer;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-06
 */
public class MLP {
    private static final Logger LOGGER = LoggerFactory.getLogger (MLP.class);
        
    public static void main (String[] args) {
        Cost cost = new Cost (new SquaredError(), 0.00, 0.0001);
        Optimizer sgd = new SGD (0.01);
        
        ByteInputLayer layer0 = new ByteInputLayer ();
        HiddenLayer    layer1 = new FullyConnectedLayer (10, new HyperbolicTangent (), cost.getLoss ());
        OutputLayer    layer2 = new OutputLayerImpl (3, new Softmax (), cost.getLoss ());
        
        String irisTrainFile = "Iris_train.data";
        
        // ClassLoader classLoader = MLP.class.getClassLoader ();
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        LOGGER.info  ("Class loader: " + classLoader.toString ());
        URL resource = classLoader.getResource (irisTrainFile);
        LOGGER.info ("Resource: " + resource);
        String irisTrainFilePath = resource.getPath();
        LOGGER.info ("irisTrainFilePath: " + irisTrainFilePath);
        
        Path irisTrainFilePathB = FileSystems.getDefault().getPath(irisTrainFile);
        LOGGER.info ("irisTrainFilePathB: " + irisTrainFilePathB);
        
        MultilayerPerceptron <InputLayer, OutputLayer> irisPerceptron = new MultilayerPerceptron <> (
            new InputLayerImpl (4),
            Arrays.asList (layer1),
            layer2,
            cost //TODO: nacpat loss do vrsvev prez vlastnost
        );
        
        try {
            IrisReader irisReader = new IrisReader ();
            
            List<Pair <double[], IrisClass>> dataset = irisReader.getDataSet(IrisData.getData(), true);
            
            irisPerceptron.initializeWeights(123456789);
            
            //TODO: u datasetu vyřeš to načítání ze souboru. Loss má teď zadrátováno že je jen pro klasifikaci, to se může zobecnit
            //      chtělo by to i vyhodnocení výsledků na testovací sadě, precission, accuracy.
            for (int epoch = 0; epoch < 50; epoch++) {
                for (Pair<double[], IrisClass> entry : dataset) {
                    double[] attributes = entry.getA ();
                    int classNumber = entry.getB ().ordinal ();

                    irisPerceptron.getInputLayer ().setInput (attributes);
                    irisPerceptron.forwardPass ();
                    irisPerceptron.setExpectedOutput (classNumber);
                    
                    System.out.println ("Iris output: " + Arrays.toString (irisPerceptron.getOutput ()) + ", expected: " + classNumber);
                    //TODO: chtělo by to vypsat error celé jedné dávky (metoda je na to připdavena v Cost)
                    
                    irisPerceptron.backwardPass ();
                    irisPerceptron.adaptWeights (sgd);            
                }
            }
        }
        catch (Exception exception) {
            LOGGER.error ("Iris test failed.", exception);
        }
    }
}
