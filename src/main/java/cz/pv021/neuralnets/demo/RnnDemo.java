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
import cz.pv021.neuralnets.utils.ModelStatistics;
import java.io.IOException;
import java.nio.file.Files;
import java.util.LinkedList;
import java.util.Objects;

/**
 * @author  Josef Plch
 * @since   2016-12-13
 * @version 2016-12-16
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
        double learningRate = 0.1; // 0.01;
        double momentum = 0.0;
        double l1 = 0.00;
        double l2 = 0.0001;
        int classes = UdLanguage.size ();
        
        Cost cost = new Cost (new MeanSquaredError(), l1, l2);
        Optimizer optimizer = new Optimizer (learningRate, new SGD(), momentum, l1, l2);
        
        Initializer initializer = new Initializer (new NormalInitialization (123456));
        
        final int INPUT_SIZE = 256;
        InputLayer  layer0 = new InputLayerImpl (0, INPUT_SIZE);
        HiddenLayer layer1 = new FullyConnectedLayer (1, 16, new HyperbolicTangent ());
        OutputLayer layer2 = new OutputLayerImpl (2, classes, new Softmax ());
        
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
        List <String> czechSentences = Files.readAllLines (csDataFilePath, StandardCharsets.UTF_8);
        LOGGER.info ("Train set size: " + czechSentences.size ());

        int[][] confusionMatrix = new int[classes][classes];
        for (int row = 0; row < classes; row++) {
            for (int col = 0; col < classes; col++) {
                confusionMatrix[row][col] = 0;
            }
        }
        
        int i = 1;
        for (String sentence : czechSentences.subList (0, 5)) {
            System.out.println ();
            System.out.println ("=== Sentence #" + i + " ===");
            System.out.println ("Sentence: " + sentence);
            
            byte[] bytes = sentence.getBytes (StandardCharsets.UTF_8);
            List <double[]> attributeSequence = new LinkedList <> ();
            for (byte byte8 : bytes) {
                attributeSequence.add (ByteUtils.byteToOneHotVector (byte8));
            }
            
            /*
            // Alternative (test) dataset.
            double[] letterH = new double[4];
            letterH[0] = 1; letterH[1] = 0; letterH[2] = 0; letterH[3] = 0;
            
            double[] letterE = new double[4];
            letterE[0] = 0; letterE[1] = 1; letterE[2] = 0; letterE[3] = 0;
            
            double[] letterL = new double[4];
            letterL[0] = 0; letterL[1] = 0; letterL[2] = 1; letterL[3] = 0;
            
            double[] letterO = new double[4];
            letterO[0] = 0; letterO[1] = 0; letterO[2] = 0; letterO[3] = 1;
            
            attributeSequence = new LinkedList <> ();
            for (int i2 = 0; i2 < 10; i2++) {
                attributeSequence.add (letterH);
                attributeSequence.add (letterE);
                attributeSequence.add (letterL);
                attributeSequence.add (letterL);
                attributeSequence.add (letterO);
            }
            */
            
            HiddenLayer hiddenLayerBefore = rnn.getHiddenLayers ().get (0);
            String weightsBefore = ModelStatistics.show2dArray (hiddenLayerBefore.getWeights ());
            
            UdExample example = new UdExample (attributeSequence, UdLanguage.FRENCH);
            
            // LEARNING.
            final int unfoldedLayers = 3;
            rnn.backpropagationThroughTime (
                example.getAttributes (),
                example.getExampleClass().getIndex (),
                unfoldedLayers
            );
            rnn.adaptWeights ();
            
            int classNumber = example.getExampleClass ().getIndex ();
            int predictedClassNumber = rnn.getOutputClassIndex ();
            
            confusionMatrix[classNumber][predictedClassNumber]++;
            
            HiddenLayer hiddenLayer = rnn.getHiddenLayers ().get (0);
            String weightsAfter = ModelStatistics.show2dArray (hiddenLayer.getWeights ());
            // System.out.println ("HL weights:\n" + weightsAfter);
            // System.out.println ("HL id: " + hiddenLayer.getId ());
            System.out.println ("Weights before are equal to weights after: " + Objects.equals (weightsBefore, weightsAfter));
            
            /*
            System.out.println ("HL innerPotentials = " + Arrays.toString (hiddenLayer.getInnerPotentials ()));
            System.out.println ("HL gradient = " + Arrays.toString (hiddenLayer.getInnerPotentialGradient ()));
            System.out.println ("OL innerPotentials = " + Arrays.toString (rnn.getOutputLayer ().getInnerPotentials()));
            System.out.println ("OL gradient = " + Arrays.toString (rnn.getOutputLayer ().getInnerPotentialGradient()));
            */
            System.out.println ("OL output = " + Arrays.toString (rnn.getOutput ()));
            System.out.println ("predictedClass = " + predictedClassNumber);
            System.out.println ("expectedClass = " + classNumber);
            
            i++;
        }
        
        System.out.println (
            ModelStatistics.modelStatistics (
                confusionMatrix,
                UdLanguage.values ()
            )
        );
    }
}
