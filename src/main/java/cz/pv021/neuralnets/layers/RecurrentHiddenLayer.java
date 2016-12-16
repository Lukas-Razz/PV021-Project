package cz.pv021.neuralnets.layers;

/**
 * @author  Josef Plch
 * @since   2016-12-12
 * @version 2016-12-16
 */
public interface RecurrentHiddenLayer extends HiddenLayer {
    @Override
    public RecurrentHiddenLayer makeCopy (int id);
}
