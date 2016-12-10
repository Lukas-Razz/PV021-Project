package cz.pv021.neuralnets.optimizers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Implementation of AdaGrad optimization.
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-10
 */
public class AdaGrad implements OptimizerAlgorithm {
    private static final Logger LOGGER = LoggerFactory.getLogger (AdaGrad.class);
    
    private Map<Integer, LayerParameters> c_g = new HashMap<Integer, LayerParameters>();
    
    @Override
    public LayerParameters computeChange(List<LayerParameters> gradients) {        
        /*LayerParameters avgGradient = gradients.get(0);
        int batchSize = gradients.size();
        
        //Summing
        for(int k=1; k<batchSize; k++) {
            for(int i=0; i<gradients.get(k).getWeights().length; i++) {
                for(int j=0; j<gradients.get(k).getWeights()[i].length; j++) {
                    avgGradient.getWeights()[i][j] += gradients.get(k).getWeights()[i][j]; //Weights
                }
                avgGradient.getBias()[i] += gradients.get(k).getBias()[i]; //Bias
            }
        }
        //Averaging
        for(int i=0; i<avgGradient.getWeights().length; i++) {
            for(int j=0; j<avgGradient.getWeights()[i].length; j++) {
                avgGradient.getWeights()[i][j] = avgGradient.getWeights()[i][j] / batchSize;
            }
            avgGradient.getBias()[i] = avgGradient.getBias()[i] / batchSize;
        }
        
        return avgGradient;*/
        throw new NotImplementedException();
    }
}
