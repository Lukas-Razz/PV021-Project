package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.dataset.DataSet;
import cz.pv021.neuralnets.dataset.iris.IrisReader;
import cz.pv021.neuralnets.dataset.iris.IrisClass;
import cz.pv021.neuralnets.dataset.iris.IrisExample;
import cz.pv021.neuralnets.optimizers.*;
import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.initialization.*;
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
import cz.pv021.neuralnets.utils.ModelStatistics;
import cz.pv021.neuralnets.utils.OutputExample;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-14
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
        double learningRate = 0.1;
        double momentum = 0.2;
        double l1 = 0.00;
        double l2 = 0.0001;
        
        Cost cost = new Cost (new RootMeanSquaredError(), l1, l2);
        Optimizer optimizer = new Optimizer(learningRate, new AdaGrad(), momentum, l1, l2);
        
        InputLayer  layer0  = new InputLayerImpl (0, 4);
        HiddenLayer layer1a = new FullyConnectedLayer (1, 10, new HyperbolicTangent());
        HiddenLayer layer1b = new FullyConnectedLayer (2, 10, new HyperbolicTangent());
        OutputLayer layer2  = new OutputLayerImpl (3, IrisClass.size (), new Softmax ());
        
        Initializer initializer = new Initializer (new NormalInitialization (123456));
        
        MultilayerPerceptron <InputLayer, OutputLayer> irisPerceptron = new MultilayerPerceptron <> (
            Arrays.asList (layer0),
            Arrays.asList (layer1a),
            layer2,
            cost,
            optimizer
        );
        irisPerceptron.initializeWeights (initializer);
        
        String irisTrainFile = "./data/iris/Iris_train.data";
        Path irisTrainFilePath = FileSystems.getDefault().getPath (irisTrainFile);
        LOGGER.info ("irisTrainFilePath: " + irisTrainFilePath.toAbsolutePath ());
        List <String> trainData = Files.readAllLines (irisTrainFilePath, StandardCharsets.UTF_8);
        LOGGER.info ("Train set size: " + trainData.size ());
        
        String irisTestFile = "./data/iris/Iris_test.data";
        Path irisTestFilePath = FileSystems.getDefault().getPath (irisTestFile);
        LOGGER.info ("irisTestFilePath: " + irisTestFilePath.toAbsolutePath ());
        List <String> testData = Files.readAllLines (irisTestFilePath, StandardCharsets.UTF_8);
        LOGGER.info ("Test set size: " + testData.size ());
        
        IrisReader irisReader = new IrisReader ();
        // Backup version: IrisData.getData ()
        DataSet <IrisClass, IrisExample> dataSet = new DataSet <> (irisReader.readDataSet (trainData), irisReader.readDataSet (testData));
        dataSet.normalizeToMinusOnePlusOne();
        
        // TODO: Loss má teď zadrátováno že je jen pro klasifikaci, to se může zobecnit
        final int bachSize = 1;
        List<List<IrisExample>> batches = dataSet.splitToBatch(bachSize);
        for (int epoch = 0; epoch < 6; epoch++) {
            runIrisEpoch (irisPerceptron, batches, epoch);
        }
        
        testIris (irisPerceptron, dataSet.getTestSet ());
    }
    
    private static void runIrisEpoch (MultilayerPerceptron <?, ?> irisPerceptron, List<List<IrisExample>> batches, int epoch) {
        // Set up the confusion matrix.
        int classes = IrisClass.values().length;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }

        double error = 0;
        for(List<IrisExample> batch : batches) {
            List<OutputExample> batchOutput = new ArrayList<>();
            for (IrisExample example : batch) {
                double[] attributes = example.getAttributes ();
                int correctClassNumber = example.getExampleClass ().getIndex ();

                irisPerceptron.setInputs (Collections.singletonList (attributes));
                irisPerceptron.forwardPass ();
                irisPerceptron.setExpectedOutput (correctClassNumber);
                irisPerceptron.backwardPass ();

                batchOutput.add(new OutputExample(irisPerceptron.getOutput(), correctClassNumber));
                int predictedClassNumber = irisPerceptron.getOutputClassIndex ();
                confusionMatrix[correctClassNumber][predictedClassNumber]++;
                
                /*
                System.out.println (
                    "Output in epoch #" + epoch + ":"
                    + " classWeights = " + Arrays.toString (irisPerceptron.getOutput ())
                    + ", outputClass = " + predictedClassNumber
                    + ", expectedClass = " + classNumber
                );
                */
            }
            error += irisPerceptron.getBatchError(batchOutput);
            // pro batch > 1 zlobí Bias....u něj chyba roste geometrickou řadou. U vah je to v pohodě.
            irisPerceptron.adaptWeights ();
        }
        error = error / batches.size();
        
        System.out.println ("Epoch #" + epoch + ": average error = " + error);
    }
    
    private static void testIris (MultilayerPerceptron <?, ?> irisPerceptron, List<IrisExample> testSet) {
        // Set up the confusion matrix.
        int classes = IrisClass.values().length;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
         
        for (IrisExample example : testSet) {
            double[] attributes = example.getAttributes ();
            int correctClassNumber = example.getExampleClass ().getIndex ();

            irisPerceptron.setInputs (Collections.singletonList (attributes));
            irisPerceptron.forwardPass ();

            int predictedClassNumber = irisPerceptron.getOutputClassIndex ();
            confusionMatrix[correctClassNumber][predictedClassNumber]++;

            /*
            System.out.println (
                "Output in epoch #" + epoch + ":"
                + " classWeights = " + Arrays.toString (irisPerceptron.getOutput ())
                + ", outputClass = " + predictedClassNumber
                + ", expectedClass = " + classNumber
            );
            */
        }
        
        System.out.println ();
        System.out.println ("=== Model test results ===");
        String statistics = ModelStatistics.modelStatistics (confusionMatrix, IrisClass.values ());
        System.out.println (statistics);
        System.out.println ();
     }
}
