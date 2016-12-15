package cz.pv021.neuralnets.network;

import cz.pv021.neuralnets.initialization.Initializer;

/**
 * Interface of neural network.
 * 
 * @author  Lukáš Daubner
 * @since   2016-11-17
 * @version 2016-12-13
 */
public interface Network {
    public int getNumberOfUnits ();
    
    public void initializeWeights (Initializer initializer);
}
