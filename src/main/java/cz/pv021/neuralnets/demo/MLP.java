package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import cz.pv021.neuralnets.network.MultilayerPerceptron;
import cz.pv021.neuralnets.optimalizers.*;
import cz.pv021.neuralnets.utils.Pair;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-04
 */
public class MLP {
    private static final Logger LOGGER = LoggerFactory.getLogger (MLP.class);
        
    public static void main (String[] args) {
        Cost cost = new Cost (new NegativeLogLikehood(), 0.00, 0.0001);
        SGD sgd = new SGD (0.01);
        
        ByteInputLayer layer0 = new ByteInputLayer ();
        HiddenLayer    layer1 = new FullyConnectedLayer (10, new HyperbolicTangent (), cost.getLoss ());
        OutputLayer    layer2 = new OutputLayerImpl (3, new Softmax (), cost.getLoss ());
        
        /*
        double[] input = {0.5, 0.2, 0.6, 0.8};
        layer0.setInput (input);
        layer1.forwardPass ();
        layer2.forwardPass ();
        System.out.println (Arrays.toString (layer2.getOutput ()));
        */
        
        MultilayerPerceptron <ByteInputLayer, OutputLayer> perceptron = new MultilayerPerceptron <> (
            layer0,
            Arrays.asList (layer1),
            layer2
        );
        
        String sentence = "Hello world!";
        byte[] sentenceBytes = sentence.getBytes (StandardCharsets.UTF_8);
        int epochs = 1;
        long seed = 123;
        
        perceptron.initializeWeights (seed);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (byte sentenceByte : sentenceBytes) {
                int label = sentenceByte % 3;
                LOGGER.info (String.valueOf (label));

                perceptron.getInputLayer ().setInputByte (sentenceByte);
                perceptron.forwardPass ();
                perceptron.setExpectedOutput (label);
                perceptron.backwardPass ();
                perceptron.adaptWeights (sgd);

                /*
                List <LayerParameters> parameters = new ArrayList <> ();
                parameters.add (layer1.getParameters());
                parameters.add (layer2.getParameters());

                List <LayerParameters> gradients = new ArrayList <> ();
                gradients.add (layer1.getErrors().get(0));
                gradients.add (layer2.getErrors().get(0));

                layer2.setParameters (sgd.changeParameters (parameters.get(0), gradients.get(0)));
                layer2.setParameters (sgd.changeParameters (parameters.get(1), gradients.get(1)));
                */

                System.out.println (Arrays.toString (perceptron.getOutput ()));
            }
        }
        
        // ----------------------------- [ Iris ] ------------------------------
        
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
            layer2
        );
        
        try {
            IrisReader irisReader = new IrisReader ();
            
            List <String> trainData;
            // trainData = Files.readAllLines (irisTrainFilePathB, StandardCharsets.UTF_8);
            trainData = Arrays.asList (
                "5.4,3.9,1.7,0.4,Iris-setosa",
                "4.6,3.4,1.4,0.3,Iris-setosa",
                "5.7,2.8,4.5,1.3,Iris-versicolor",
                "6.3,3.3,4.7,1.6,Iris-versicolor",
                "4.9,2.5,4.5,1.7,Iris-virginica",
                "7.3,2.9,6.3,1.8,Iris-virginica"
            );
            
            // BufferedReader trainDataReader = new BufferedReader (new FileReader (irisTrainFilePath));
            // String line = trainDataReader.readLine ();
            
            for (int epoch = 0; epoch < 10; epoch++) {
                for (String line : trainData) {
                    Pair <double[], IrisClass> entry = irisReader.readEntry (line);
                    double[] attributes = entry.getA ();
                    int classNumber = entry.getB ().ordinal ();

                    irisPerceptron.initializeWeights (49);
                    irisPerceptron.getInputLayer ().setInput (attributes);
                    irisPerceptron.forwardPass ();
                    irisPerceptron.setExpectedOutput (classNumber);
                    irisPerceptron.backwardPass ();
                    irisPerceptron.adaptWeights (sgd);            

                    // line = trainDataReader.readLine ();
                }
            }
            
            System.out.println (
                "Iris output: "
                + Arrays.toString (irisPerceptron.getOutput ())
            );
        }
        catch (Exception exception) {
            LOGGER.error ("Iris test failed.", exception);
        }
        
        // -------------------------- [ End of Iris ] --------------------------
        
        /*
        List <OutputExample> batch = new ArrayList <> ();
        batch.add(new OutputExample(layer2.getOutput(), 1));
        
        // GetNetworkParameters
        List <LayerParameters> parameters = new ArrayList <> ();
        parameters.add (layer1.getParameters());
        parameters.add (layer2.getParameters());
        double error = cost.getError(batch, parameters);
        
        logger.info (String.valueOf (error));
        logger.info (Arrays.toString (layer2.getOutput ()));
        */
    }
}
