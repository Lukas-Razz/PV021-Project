package cz.pv021.neuralnets.optimizers;

import cz.pv021.neuralnets.utils.LayerParameters;
import static java.lang.Math.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of AdaGrad optimization.
 * 
 * @author  Lukáš Daubner
 * @since   2016-12-10
 * @version 2016-12-10
 */
public class AdaGrad implements OptimizerAlgorithm {
    private static final Logger LOGGER = LoggerFactory.getLogger (AdaGrad.class);
    
    private Map<Integer, LayerParameters> c_gMap = new HashMap<>();
    
    @Override
    public LayerParameters computeChange(List<LayerParameters> gradients) {        
        int batchSize = gradients.size();
        
        //Summing
        LayerParameters gradient = gradients.get(0);
        for(int k=1; k<batchSize; k++) {
            for(int i=0; i<gradients.get(k).getWeights().length; i++) {
                for(int j=0; j<gradients.get(k).getWeights()[i].length; j++) {
                    gradient.getWeights()[i][j] += gradients.get(k).getWeights()[i][j]; //Weights
                }
                gradient.getBias()[i] += gradients.get(k).getBias()[i]; //Bias
            }
        }
        
        //Get saved c_g, or initialize new
        LayerParameters c_g = null;
        if(c_gMap.containsKey(gradient.getLayerId())) {
            c_g = c_gMap.get(gradient.getLayerId());
        }
        else {
            double[][] w = new double[gradient.getWeights().length][];
            double[] b = new double[gradient.getBias().length];
            for(int i=0; i<gradient.getWeights().length; i++) {
                w[i] = new double[gradient.getWeights()[i].length];
                for(int j=0; j<gradient.getWeights()[i].length; j++) {
                    w[i][j] = 0;
                }
                b[i] = 0;
            }
            c_g = new LayerParameters(w, b, gradient.getLayerId());
        }
        
        //Compute change 
        for(int i=0; i<gradient.getWeights().length; i++) {
            for(int j=0; j<gradient.getWeights()[i].length; j++) {
                c_g.getWeights()[i][j] = c_g.getWeights()[i][j] + pow(gradient.getWeights()[i][j], 2);
                gradient.getWeights()[i][j] = gradient.getWeights()[i][j] / sqrt(c_g.getWeights()[i][j]);
            }
            c_g.getBias()[i] = c_g.getBias()[i] + pow(gradient.getBias()[i], 2);
            gradient.getBias()[i] = gradient.getBias()[i] / sqrt(c_g.getBias()[i]);
        }
        
        c_gMap.put(gradient.getLayerId(), c_g);
        return gradient;
    }
}
