package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;import cz.pv021.neuralnets.utils.LayerParameters;
import cz.pv021.neuralnets.utils.OutputExample;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-11-21
 */
public class MLP {
    public static void main (String[] args) {
        final Logger logger = LoggerFactory.getLogger (MLP.class);
        
        ByteInputLayer layer0 = new ByteInputLayer ();
        HiddenLayer    layer1 = new FullyConnectedLayer (10, new HyperbolicTangent ());
        OutputLayer    layer2 = new OutputLayerImpl (3, new Softmax ());
        Layers.connect (layer0, layer1);
        Layers.connect (layer1, layer2);

        long seed = 123;
        layer1.initializeWeights (seed);
        layer2.initializeWeights (seed);
        
        /*
        double[] input = {0.5, 0.2, 0.6, 0.8};
        layer0.setInput (input);
        layer1.forwardPass ();
        layer2.forwardPass ();
        System.out.println (Arrays.toString (layer2.getOutput ()));
        */
        
        String sentence = "Hello world!";
        try {
            byte[] sentenceBytes = sentence.getBytes ("UTF-8");
            for (byte sentenceByte : sentenceBytes) {
                layer0.setInputByte (sentenceByte);
                layer1.forwardPass ();
                layer2.forwardPass ();
            }
        }
        catch (UnsupportedEncodingException exception) {
            System.out.println (exception);
        }
        
        System.out.println (Arrays.toString (layer2.getOutput ()));
        
        List <OutputExample> batch = new ArrayList <> ();
        batch.add(new OutputExample(layer2.getOutput(), 1));
        
        // GetNetworkParameters
        List <LayerParameters> parameters = new ArrayList <> ();
        parameters.add (layer1.getParameters());
        parameters.add (layer2.getParameters());
        Cost cost = new Cost (new NegativeLogLikehood(), 0.00, 0.0001);
        double error = cost.getError(batch, parameters);
        
        logger.info (String.valueOf (error));
        logger.info (Arrays.toString (layer2.getOutput ()));
    }
}
