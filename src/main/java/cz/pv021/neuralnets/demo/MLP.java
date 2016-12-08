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
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-08
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
        Optimizer optimizer = new Optimizer(0.01, new SGD());
        
        InputLayer  layer0  = new InputLayerImpl (4);
        HiddenLayer layer1a = new FullyConnectedLayer (10, new HyperbolicTangent ());
        HiddenLayer layer1b = new FullyConnectedLayer (10, new HyperbolicTangent ());
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
        
        IrisReader irisReader = new IrisReader ();
        // Backup version: IrisData.getData ()
        List <IrisExample> dataset = irisReader.readDataSet (trainData, true);

        DecimalFormat formatter = new DecimalFormat ("#.0");
        DecimalFormatSymbols formatSymbols = new  DecimalFormatSymbols ();
        formatSymbols.setDecimalSeparator ('.');
        formatter.setDecimalFormatSymbols (formatSymbols);
        
        // TODO: Loss má teď zadrátováno že je jen pro klasifikaci, to se může zobecnit
        final int bachSize = 1;
        for (int epoch = 0; epoch < 50; epoch++) {
            runIrisEpoch (irisPerceptron, dataset, formatter, bachSize, epoch);
        }
    }
    
    private static void runIrisEpoch (MultilayerPerceptron <InputLayer, OutputLayer> irisPerceptron, List <IrisExample> dataset, DecimalFormat formatter, int batchSize, int epoch) {
        // Set up the confusion matrix.
        int classes = IrisClass.values().length;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }

        int batchIter = 0;
        for (IrisExample example : dataset) {
            double[] attributes = example.getAttributes ();
            int correctClassNumber = example.getIrisClass ().ordinal ();

            irisPerceptron.getInputLayer ().setInput (attributes);
            irisPerceptron.forwardPass ();
            irisPerceptron.setExpectedOutput (correctClassNumber);
            irisPerceptron.backwardPass ();

            // pro batch > 1 zlobí Bias....u něj chyba roste geometrickou řadou. U vah je to v pohodě.
            if (batchIter == batchSize - 1) {
                irisPerceptron.adaptWeights ();
                batchIter = 0;
            }
            else {
                batchIter++;
            }

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
        
        System.out.println ("=== Epoch #" + epoch + " ===");
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
}
