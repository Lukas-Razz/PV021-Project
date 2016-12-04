package cz.pv021.neuralnets.utils;

import java.util.Objects;

/**
 * An ordered pair.
 * 
 * @author  Josef Plch
 * @since   2016-12-03
 * @version 2016-12-04
 */
public class Pair <A, B> {
    private final A a;
    private final B b;
    
    public Pair (A a, B b) {
        this.a = a;
        this.b = b;
    }
    
    public A getA () {
        return a;
    }
    
    public B getB () {
        return b;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 79 * hash + Objects.hashCode (this.a);
        hash = 79 * hash + Objects.hashCode (this.b);
        return hash;
    }

    @Override
    public boolean equals (Object object) {
        boolean result;
        if (object == null || ! (object instanceof Pair)) {
            result = false;
        }
        else {
            final Pair <?, ?> other = (Pair <?, ?>) object;
            result = Objects.equals (this.a, other.a) && Objects.equals (this.b, other.b);
        }
        return result;
    }
}
