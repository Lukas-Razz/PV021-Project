package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.dataset.DataSet;
import cz.pv021.neuralnets.dataset.Example;
import cz.pv021.neuralnets.dataset.iris.IrisReader;
import cz.pv021.neuralnets.dataset.iris.IrisClass;
import cz.pv021.neuralnets.optimizers.SGD;
import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import cz.pv021.neuralnets.network.MultilayerPerceptron;
import cz.pv021.neuralnets.network.RecurrentNetwork;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cz.pv021.neuralnets.optimizers.Optimizer;
import cz.pv021.neuralnets.utils.OutputExample;
import java.io.IOException;
import java.nio.file.Files;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.LinkedList;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-10
 */
public class MLP {
    private static final Logger LOGGER = LoggerFactory.getLogger (MLP.class);
        
    public static void main (String[] args) {
        try {
            testIris ();
            testSentences ();
        }
        catch (IOException exception) {
            LOGGER.error ("Iris test failed.", exception);
        }
    }
    
    private static void testIris () throws IOException {
        double learningRate = 0.01;
        double l1 = 0.00;
        double l2 = 0.0001;
        
        Cost cost = new Cost (new SquaredError(), l1, l2);
        Optimizer optimizer = new Optimizer(learningRate, new SGD(), l1, l2);
        
        InputLayer  layer0  = new InputLayerImpl (4);
        HiddenLayer layer1a = new FullyConnectedLayer (10, new HyperbolicTangent());
        HiddenLayer layer1b = new FullyConnectedLayer (10, new HyperbolicTangent());
        OutputLayer layer2  = new OutputLayerImpl (3, new Softmax ());
        
        MultilayerPerceptron <InputLayer, OutputLayer> irisPerceptron = new MultilayerPerceptron <> (
            layer0,
            Arrays.asList (layer1a),
            layer2,
            cost,
            optimizer
        );
        irisPerceptron.initializeWeights (123456789);
        
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
        DataSet dataSet = new DataSet(irisReader.readDataSet (trainData), irisReader.readDataSet (testData));
        dataSet.normalizeToMinusOnePlusOne();
        
        DecimalFormat formatter = new DecimalFormat ("#.0");
        DecimalFormatSymbols formatSymbols = new  DecimalFormatSymbols ();
        formatSymbols.setDecimalSeparator ('.');
        formatter.setDecimalFormatSymbols (formatSymbols);
        
        // TODO: Loss má teď zadrátováno že je jen pro klasifikaci, to se může zobecnit
        final int bachSize = 1;
        List<List<Example>> batches = dataSet.splitToBatch(bachSize);
        for (int epoch = 0; epoch < 50; epoch++) {
            runIrisEpoch (irisPerceptron, batches, formatter, epoch);
        }
        
        testIris(irisPerceptron, dataSet.getTestSet(), formatter);
    }
    
    private static void runIrisEpoch (MultilayerPerceptron <InputLayer, OutputLayer> irisPerceptron, List<List<Example>> batches, DecimalFormat formatter, int epoch) {
        // Set up the confusion matrix.
        int classes = IrisClass.values().length;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }

        double error = 0;
        for(List<Example> batch : batches) {
            List<OutputExample> batchOutput = new ArrayList<>();
            for (Example example : batch) {
                double[] attributes = example.getAttributes ();
                int correctClassNumber = example.getIrisClass ().ordinal ();

                irisPerceptron.getInputLayer ().setInput (attributes);
                irisPerceptron.forwardPass ();
                irisPerceptron.setExpectedOutput (correctClassNumber);
                irisPerceptron.backwardPass ();

                batchOutput.add(new OutputExample(irisPerceptron.getOutput(), correctClassNumber));
                int predictedClassNumber = irisPerceptron.getOutputClassIndex ();
                confusionMatrix[correctClassNumber][predictedClassNumber]++;
                
                // TODO: chtělo by to vypsat error celé jedné dávky (metoda je na to připdavena v Cost)
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
        
        System.out.println ("=== Epoch #" + epoch + " ===");
        System.out.println ("Average error: " + error);
        System.out.println ();
        /*
        String statistics = modelStatistics (confusionMatrix, formatter);
        System.out.println (statistics);
        System.out.println ();
        */
    }
    
     private static void testIris (MultilayerPerceptron <InputLayer, OutputLayer> irisPerceptron, List<Example> testSet, DecimalFormat formatter) {
        // Set up the confusion matrix.
        int classes = IrisClass.values().length;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
         
        for (Example example : testSet) {
            double[] attributes = example.getAttributes ();
            int correctClassNumber = example.getIrisClass ().ordinal ();

            irisPerceptron.getInputLayer ().setInput (attributes);
            irisPerceptron.forwardPass ();

            int predictedClassNumber = irisPerceptron.getOutputClassIndex ();
            confusionMatrix[correctClassNumber][predictedClassNumber]++;

            // TODO: chtělo by to vypsat error celé jedné dávky (metoda je na to připdavena v Cost)
            /*
            System.out.println (
                "Output in epoch #" + epoch + ":"
                + " classWeights = " + Arrays.toString (irisPerceptron.getOutput ())
                + ", outputClass = " + predictedClassNumber
                + ", expectedClass = " + classNumber
            );
            */
        }
        
        System.out.println ("=== Model test results ===");
        String statistics = modelStatistics (confusionMatrix, formatter);
        System.out.println (statistics);
        System.out.println ();
     }
    
    /**
     * Show model statistics.
     * 
     * @param confusionMatrix It must be square matrix.
     * @param epoch           Number of epoch.
     * @param formatter       Formatter of decimal numbers.
     * @return                Serialized statistics.
     */
    private static String modelStatistics (int[][] confusionMatrix, DecimalFormat formatter) {
        int classes = confusionMatrix.length;
        
        int totalInstances = 0;
        int correct = 0;
        for (int x = 0; x < classes; x++) {
            correct += confusionMatrix[x][x];
            for (int y = 0; y < classes; y++) {
                totalInstances += confusionMatrix[x][y];
            }
        }
        double overallAccuracy = 100.0 * correct / totalInstances;
        
        // Overall statistics.
        StringBuilder result = new StringBuilder ();
        result
            .append ("Correctly Classified Instances  \t")
            .append (correct).append("\t").append(formatter.format (overallAccuracy)).append (" %")
            .append ("\nIncorrectly Classified Instances\t")
            .append (totalInstances - correct).append("\t").append(formatter.format (100 - overallAccuracy)).append (" %")
            .append ("\nTotal Number of Instances       \t")
            .append (totalInstances)
            .append ("\n");
        
        // Confusion matrix heading.
        IrisClass[] classArray = IrisClass.values();
        result.append ("\nPredicted ->");
        for (int x = 0; x < classes; x++) {
            result.append ("\t").append (x);
        }
        result.append ("\tPrec.").append ("\tRecall");
        
        result.append ("\n--------------------------------------------------------");
        
        // Confusion matrix body.
        for (int x = 0; x < classes; x++) {
            // Class index: class name
            result.append ("\n").append (x).append (": ").append (classArray[x].name ());
            
            int xAsX = confusionMatrix[x][x];
            int xAsAny = 0;
            int anyAsX = 0;
            for (int y = 0; y < classes; y++) {
                int xAsY = confusionMatrix[x][y];
                int yAsX = confusionMatrix[y][x];
                xAsAny += xAsY;
                anyAsX += yAsX;
                result.append ("\t").append (xAsY);
            }
            
            // Precision = TP / (TP + FP)
            double classPrecision = 100.0 * xAsX / anyAsX;
            result.append ("\t").append (formatter.format (classPrecision));
            
            // Recall = TP / (TP + FN)
            double classRecall = 100.0 * xAsX / xAsAny;
            result.append("\t").append (formatter.format (classRecall));
        }
        
        return result.toString ();
    }
    
    private static void testSentences () throws IOException {
        double learningRate = 0.01;
        double l1 = 0.00;
        double l2 = 0.0001;
        
        Cost cost = new Cost (new SquaredError(), l1, l2);
        Optimizer optimizer = new Optimizer(learningRate, new SGD(), l1, l2);
        
        InputLayer  layer0  = new InputLayerImpl (256);
        HiddenLayer layer1a = new FullyConnecedRecursiveLayer (1, new HyperbolicTangent());
        // HiddenLayer layer1b = new FullyConnectedLayer (10, new HyperbolicTangent());
        OutputLayer layer2  = new OutputLayerImpl (10, new Softmax ());
        
        RecurrentNetwork <InputLayer, OutputLayer> irisPerceptron = new RecurrentNetwork <> (
            layer0,
            Arrays.asList (layer1a),
            layer2,
            cost,
            optimizer
        );
        irisPerceptron.initializeWeights (123456789);
        
        String csDataPath = "./data/language_identification/cs_sentences.txt";
        Path csDataFilePath = FileSystems.getDefault().getPath (csDataPath);
        LOGGER.info ("irisTrainFilePath: " + csDataFilePath.toAbsolutePath ());
        List <String> csSentences = Files.readAllLines (csDataFilePath, StandardCharsets.UTF_8);
        LOGGER.info ("Train set size: " + csSentences.size ());

        int classes = 2;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
         
        for (String csSentence : csSentences) {
            byte[] bytes = csSentence.getBytes (StandardCharsets.UTF_8);
            List <double[]> attributeSequence = new LinkedList <> ();
            for (byte byte8 : bytes) {
                attributeSequence.add (ByteInputLayer.byteToDoubleArray (byte8));
            }
            int csClassNumber = 0;
            
            irisPerceptron.backpropagationThroughTime (attributeSequence, csClassNumber);
            int predictedClassNumber = irisPerceptron.getOutputClassIndex ();
            confusionMatrix[csClassNumber][predictedClassNumber]++;

            System.out.println (
                "Output in epoch #" + "?" + ":"
                + " classWeights = " + Arrays.toString (irisPerceptron.getOutput ())
                + ", outputClass = " + predictedClassNumber
                + ", expectedClass = " + csClassNumber
            );
        }
    }
}
