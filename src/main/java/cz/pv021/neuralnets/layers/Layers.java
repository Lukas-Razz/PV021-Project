package cz.pv021.neuralnets.layers;

/**
 * Podpůrná třída s pomocnými metodami pro lepší práci s vrstvami.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-07
 */
public abstract class Layers {
    /**
     * Propojení dvou vrstev.
     * 
     * @param upper Upper layer.
     * @param lower Lower layer.
     */
    public static void connect (LayerWithOutput upper, LayerWithInput lower) {
        lower.setUpperLayer (upper);
        upper.setLowerLayer (lower);
    }
}
