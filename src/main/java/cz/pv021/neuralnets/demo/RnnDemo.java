package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.dataset.udcorpora.UdExample;
import cz.pv021.neuralnets.dataset.udcorpora.UdLanguage;
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
        
        InputLayer                   layer0 = new InputLayerImpl (0, 256);
        FullyConnectedRecursiveLayer layer1 = new FullyConnectedRecursiveLayer (1, 3, new HyperbolicTangent());
        OutputLayer                  layer2 = new OutputLayerImpl (2, UdLanguage.size (), new Softmax ());
        
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
                attributeSequence.add (ByteUtils.byteToOneHotVector (byte8));
            }
            
            /*
            double[] e1 = new double [4];
            e1[0]=0.111;
            e1[1]=0.222;
            e1[2]=0.333;
            e1[3]=0.444;
            
            double[] e2 = new double [4];
            e2[0]=0.555;
            e2[1]=0.666;
            e2[2]=0.777;
            e2[3]=0.888;
            
            attributeSequence = new LinkedList <> ();
            for (int i2 = 0; i2 < 10; i2++) {
                if (true || i2 % 2 == 0) {
                    attributeSequence.add (e1);
                }
                else {
                    attributeSequence.add (e2);
                }
            }
            */
            
            UdExample example = new UdExample (attributeSequence, UdLanguage.FRENCH);
            /*
            for (UdLanguage language : UdLanguage.values()) {
                System.out.println ("Language #" + language.getIndex () + ": " + language.getName ());
            }
            */
            rnn.backpropagationThroughTime (
                example.getAttributes (),
                example.getExampleClass().getIndex ()
            );
            
            int classNumber = example.getExampleClass ().getIndex ();
            int predictedClassNumber = rnn.getOutputClassIndex ();
            confusionMatrix[classNumber][predictedClassNumber]++;
            
            System.out.println ();
            System.out.println ("=== Example #" + i + " ===");
            List <HiddenLayer> hiddenLayers = rnn.getHiddenLayers ();
            System.out.println ("HL innerPotentials = " + Arrays.toString (hiddenLayers.get(0).getInnerPotentials()));
            System.out.println ("HL gradient = " + Arrays.toString (hiddenLayers.get(0).getInnerPotentialGradient()));
            System.out.println ("OL innerPotentials = " + Arrays.toString (rnn.getOutputLayer ().getInnerPotentials()));
            System.out.println ("OL gradient = " + Arrays.toString (rnn.getOutputLayer ().getInnerPotentialGradient()));
            System.out.println ("OL output = " + Arrays.toString (rnn.getOutput ()));
            System.out.println ("predictedClass = " + predictedClassNumber);
            System.out.println ("expectedClass = " + classNumber);
            
            i++;
        }
    }
}
