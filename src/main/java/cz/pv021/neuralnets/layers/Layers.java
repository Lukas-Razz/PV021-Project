package cz.pv021.neuralnets.layers;

/**
 * Helper class for layers
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-07
 */
public abstract class Layers {
    /**
     * Connect two layers.
     * 
     * @param upper Upper layer.
     * @param lower Lower layer.
     */
    public static void connect (LayerWithOutput upper, LayerWithInput lower) {
        lower.setUpperLayer (upper);
        upper.setLowerLayer (lower);
    }
}
