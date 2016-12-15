package cz.pv021.neuralnets.layers;

/**
 * @author  Josef Plch
 * @since   2016-12-12
 * @version 2016-12-14
 */
public interface RecurrentHiddenLayer extends HiddenLayer {
    public HiddenLayer makeNonRecurrentCopy (int id);
}
