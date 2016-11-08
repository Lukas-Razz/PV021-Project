package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import java.util.Arrays;

/**
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-08
 */
public class MLP {
    public static void main (String[] args) {
        InputLayer  layer0 = new InputLayerImpl (4);
        HiddenLayer layer1 = new FullyConnectedLayer (10, new HyperbolicTangent ());
        OutputLayer layer2 = new OutputLayerImpl (3, new Softmax ());
        Layers.connect (layer0, layer1);
        Layers.connect (layer1, layer2);

        long seed = 123;
        layer1.initializeWeights (seed);
        layer2.initializeWeights (seed);

        double[] input = {0.5, 0.2, 0.6, 0.8};
        layer0.clampInput (input);
        layer1.forwardPass ();
        layer2.forwardPass ();
        System.out.println (Arrays.toString (layer2.getOutput ()));
    }
}
