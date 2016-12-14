package cz.pv021.neuralnets.layers;

/**
 * The most general interface for all layers.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-14
 */
public interface Layer {
    public int getNumberOfUnits ();
    
    public double [] getOutput ();
    
    public int getId();
}

