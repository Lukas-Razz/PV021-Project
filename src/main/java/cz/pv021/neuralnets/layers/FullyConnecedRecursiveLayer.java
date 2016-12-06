package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-06
 */
public class FullyConnecedRecursiveLayer implements HiddenLayer {
    final Logger logger = LoggerFactory.getLogger(FullyConnecedRecursiveLayer.class);
    
    @Override
    public void backwardPass () {
        throw new UnsupportedOperationException ("Not supported yet.");
    }

    @Override
    public void forwardPass () {
        throw new UnsupportedOperationException ("Not supported yet.");
    }

    @Override
    public int getNumberOfUnits () {
        throw new UnsupportedOperationException ("Not supported yet.");
    }

    @Override
    public double[] getOutput () {
        throw new UnsupportedOperationException ("Not supported yet.");
    }

    @Override
    public void initializeWeights (long seed) {
        throw new UnsupportedOperationException ("Not supported yet.");
    }
    
    @Override
    public void setLowerLayer (LayerWithInput layer) {
        throw new UnsupportedOperationException ("Not supported yet.");
    }

    @Override
    public void setUpperLayer (LayerWithOutput layer) {
        throw new UnsupportedOperationException ("Not supported yet.");
    }

    @Override
    public LayerParameters getParameters() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public void setParameters(LayerParameters parameters) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double[] getInnerPotentialGradient() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public List<LayerParameters> getErrors() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void resetGradients() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
