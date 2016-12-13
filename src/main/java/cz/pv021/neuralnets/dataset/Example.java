package cz.pv021.neuralnets.dataset;

/**
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-12-10
 * @version 2016-12-13
 */
public interface Example {
    double[] getAttributes ();

    DataClass getExampleClass ();
}
