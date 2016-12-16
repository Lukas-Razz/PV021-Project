package cz.pv021.neuralnets.error;

import cz.pv021.neuralnets.utils.LayerParameters;
import cz.pv021.neuralnets.utils.OutputExample;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * @author Lukas Daubner
 * @since   2016-11-17
 * @version 2016-12-06
 */
public class Cost {
    final Logger logger = LoggerFactory.getLogger(Cost.class);
    
    private Loss loss;
    
    private double l1;
    private double l2;
    
    public Cost (Loss loss, double l1, double l2) {
        this.loss = loss;
        this.l1 = l1;
        this.l2 = l2;
    }
    
    public Loss getLoss()
    {
        return loss;
    }
    
    public double getBatchError(List<OutputExample> batch, List<LayerParameters> parameters) {
        double lossSum = 0;
        for(OutputExample example : batch) {
            for (int i=0; i<example.getActualOutput().length; i++) {
                //pouze pro klasifikaci
                if(i == example.getExpectedOutput()) {
                    lossSum += loss.loss(example.getActualOutput()[i], 1);
                }
                else
                    lossSum += loss.loss(example.getActualOutput()[i], 0);
            }
        }
        double cost = lossSum / batch.size() 
                + (l1 != 0 ? computeL1(parameters) * l1 : 0) 
                + (l2 != 0 ? computeL2(parameters) * l2 : 0);
        return cost;
    }
    
    public double computeL1(List<LayerParameters> parameters) {
        double sum = 0;
        for(LayerParameters layer : parameters) {
            for(int i=0; i<layer.getWeights().length; i++){
                for(int j=0; j<layer.getWeights()[i].length; j++){
                    sum += Math.abs(layer.getWeights()[i][j]);
                }
            }
        }
        return sum;
    }
    
    public double computeL2(List<LayerParameters> parameters) {
        double sum = 0;
        for(LayerParameters layer : parameters) {
            for(int i=0; i<layer.getWeights().length; i++){
                for(int j=0; j<layer.getWeights()[i].length; j++){
                    sum += Math.pow(layer.getWeights()[i][j], 2);
                }
            }
        }
        return sum;
    }
    
}
