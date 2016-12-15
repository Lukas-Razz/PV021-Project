package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.optimizers.SGD;
import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import cz.pv021.neuralnets.initialization.Initializer;
import cz.pv021.neuralnets.initialization.NormalInitialization;
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
        double momentum = 0.0;
        double l1 = 0.00;
        double l2 = 0.0001;
        
        Cost cost = new Cost (new MeanSquaredError(), l1, l2);
        Optimizer optimizer = new Optimizer(learningRate, new SGD(), momentum, l1, l2);
        
        Initializer initializer = new Initializer (new NormalInitialization (123456));
        
        InputLayer                   layer0 = new InputLayerImpl (0, 4);
        // InputLayer                   layer0 = new ByteInputLayer (0);
        FullyConnectedRecursiveLayer layer1 = new FullyConnectedRecursiveLayer (1, 3, new HyperbolicTangent());
        OutputLayer                  layer2 = new OutputLayerImpl (2, 4, new Softmax ());
        
        RecurrentNetwork <InputLayer, OutputLayer> rnn = new RecurrentNetwork <> (
            layer0,
            layer1,
            layer2,
            cost,
            optimizer
        );
        rnn.initializeWeights (initializer);
        
        String csDataPath = "./data/language_identification/cs_sentences.txt";
        Path csDataFilePath = FileSystems.getDefault().getPath (csDataPath);
        LOGGER.info ("irisTrainFilePath: " + csDataFilePath.toAbsolutePath ());
        List <String> csSentences = Files.readAllLines (csDataFilePath, StandardCharsets.UTF_8);
        LOGGER.info ("Train set size: " + csSentences.size ());

        int classes = 5;
        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
        
        int i = 1;
        for (String csSentence : csSentences.subList (0, 2)) {
            byte[] bytes = csSentence.getBytes (StandardCharsets.UTF_8);
            List <double[]> attributeSequence = new LinkedList <> ();
            for (byte byte8 : bytes) {
                double[] x = new double [4];
                x[0]=0.111;
                x[1]=0.222;
                x[2]=0.333;
                x[3]=0.444;
                attributeSequence.add (x);
                // attributeSequence.add (ByteUtils.byteToOneHotVector (byte8));
            }
            
            int classNumber = 2;
            rnn.backpropagationThroughTime (attributeSequence, classNumber);
            int predictedClassNumber = rnn.getOutputClassIndex ();
            confusionMatrix[classNumber][predictedClassNumber]++;
            
            System.out.println ();
            System.out.println ("=== Example #" + i + " ===");
            List <HiddenLayer> hiddenLayers = rnn.getHiddenLayers ();
            System.out.println ("innerPotentials = " + Arrays.toString (hiddenLayers.get(0).getInnerPotentials()));
            System.out.println (
                "output = " + Arrays.toString (rnn.getOutput ())
                // + ", predictedClass = " + predictedClassNumber
                // + ", expectedClass = " + classNumber
            );
            
            i++;
        }
    }
}
