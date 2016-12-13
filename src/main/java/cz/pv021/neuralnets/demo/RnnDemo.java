package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.optimizers.SGD;
import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import cz.pv021.neuralnets.network.RecurrentNetwork;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import cz.pv021.neuralnets.optimizers.Optimizer;
import cz.pv021.neuralnets.utils.ByteUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.util.LinkedList;

/**
 * @author  Josef Plch
 * @since   2016-12-13
 * @version 2016-12-13
 */
public class RnnDemo {
    private static final Logger LOGGER = LoggerFactory.getLogger (RnnDemo.class);
        
    public static void main (String[] args) {
        try {
            testSentences ();
        }
        catch (IOException exception) {
            LOGGER.error ("The test failed.", exception);
        }
    }
    
    private static void testSentences () throws IOException {
        double learningRate = 0.01;
        double l1 = 0.00;
        double l2 = 0.0001;
        
        Cost cost = new Cost (new SquaredError(), l1, l2);
        Optimizer optimizer = new Optimizer(learningRate, new SGD(), l1, l2);
        
        InputLayer  layer0  = new InputLayerImpl (256);
        HiddenLayer layer1a = new FullyConnectedRecursiveLayer (1, new HyperbolicTangent());
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
                attributeSequence.add (ByteUtils.byteToOneHotVector (byte8));
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
