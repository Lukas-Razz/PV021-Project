package cz.pv021.neuralnets.layers;

/**
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-07
 */
public class FullyConnecedRecursiveLayer implements HiddenLayer {
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
}
