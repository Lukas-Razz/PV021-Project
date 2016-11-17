package cz.pv021.neuralnets.demo;

import cz.pv021.neuralnets.error.*;
import cz.pv021.neuralnets.layers.*;
import cz.pv021.neuralnets.functions.*;
import cz.pv021.neuralnets.utils.LayerParameters;
import cz.pv021.neuralnets.utils.OutputExample;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-17
 */
public class MLP {
    public static void main (String[] args) {
        final Logger logger = LoggerFactory.getLogger(MLP.class);
        
        Cost cost = new Cost(new NegativeLogLikehood(), 0.00, 0.0001);
        
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
        
        List<OutputExample> batch = new ArrayList<>();
        batch.add(new OutputExample(layer2.getOutput(), 1));
        
        //GetNetworkParameters
        List<LayerParameters> parameters = new ArrayList<>();
        parameters.add(layer1.getParameters());
        parameters.add(layer2.getParameters());
        //GetNetworkParameters
        
        double error = cost.getError(batch, parameters);
        
        logger.info(String.valueOf(error));
        logger.info(Arrays.toString (layer2.getOutput()));
    }
}
