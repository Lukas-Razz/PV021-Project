package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.optimizers.SGD;
import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import cz.pv021.neuralnets.network.MultilayerPerceptron;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cz.pv021.neuralnets.optimizers.Optimizer;
import java.io.IOException;
import java.nio.file.Files;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-06
 */
public class MLP {
    private static final Logger LOGGER = LoggerFactory.getLogger (MLP.class);
        
    public static void main (String[] args) {
        try {
            testIris ();
        }
        catch (IOException exception) {
            LOGGER.error ("Iris test failed.", exception);
        }
    }
    
    private static void testIris () throws IOException {
        Cost cost = new Cost (new SquaredError(), 0.00, 0.0001);
        Optimizer sgd = new SGD (0.01);
        
        ByteInputLayer layer0 = new ByteInputLayer ();
        HiddenLayer    layer1 = new FullyConnectedLayer (10, new HyperbolicTangent (), cost.getLoss ());
        OutputLayer    layer2 = new OutputLayerImpl (3, new Softmax (), cost.getLoss ());
        
        String irisTrainFile = "./data/iris/Iris_train.data";
        Path irisTrainFilePath = FileSystems.getDefault().getPath (irisTrainFile);
        LOGGER.info ("irisTrainFilePath: " + irisTrainFilePath.toAbsolutePath ());
        List <String> trainData = Files.readAllLines (irisTrainFilePath, StandardCharsets.UTF_8);
        LOGGER.info ("Train set size: " + trainData.size ());
        
        MultilayerPerceptron <InputLayer, OutputLayer> irisPerceptron = new MultilayerPerceptron <> (
            new InputLayerImpl (4),
            Arrays.asList (layer1),
            layer2,
            cost //TODO: nacpat loss do vrsvev prez vlastnost
        );
        
        IrisReader irisReader = new IrisReader ();
        // Backup version: IrisData.getData ()
        List <IrisExample> dataset = irisReader.readDataSet (trainData, true);

        irisPerceptron.initializeWeights(123456789);

        // TODO:
        // Načítání datasetu ze souboru - OK
        // Loss má teď zadrátováno že je jen pro klasifikaci, to se může zobecnit
        for (int epoch = 0; epoch < 50; epoch++) {
            int predictions = 0;
            int correctPredictions = 0;
            
            for (IrisExample example : dataset) {
                double[] attributes = example.getAttributes ();
                int classNumber = example.getIrisClass ().ordinal ();

                irisPerceptron.getInputLayer ().setInput (attributes);
                irisPerceptron.forwardPass ();
                irisPerceptron.setExpectedOutput (classNumber);
                irisPerceptron.backwardPass ();
                irisPerceptron.adaptWeights (sgd);            

                //TODO: chtělo by to vypsat error celé jedné dávky (metoda je na to připdavena v Cost)
                
                int predictedClassNumber = irisPerceptron.getOutputClassIndex ();
                /*
                System.out.println (
                    "Output in epoch #" + epoch + ":"
                    + " classWeights = " + Arrays.toString (irisPerceptron.getOutput ())
                    + ", outputClass = " + predictedClassNumber
                    + ", expectedClass = " + classNumber
                );
                */
                
                if (predictedClassNumber == classNumber) {
                    correctPredictions++;
                }
                predictions++;
            }
            
            // TODO: Chtělo by to i vyhodnocení výsledků na testovací sadě, precission, accuracy.
            System.out.println (
                "Correct predictions in epoch #" + epoch + ":"
                + " " + correctPredictions + " / " + predictions
                + " (" + (100 * 10 * correctPredictions / predictions / 10.0) + " %)"
            );
        }
    }
}
