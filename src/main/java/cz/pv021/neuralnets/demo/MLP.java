package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-11-21
 */
public class MLP {
    public static void main (String[] args) {
        // InputLayer     layer0 = new InputLayerImpl (4);
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
    }
}
