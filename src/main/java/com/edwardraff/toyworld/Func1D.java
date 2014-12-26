/*
 * Copyright (C) 2014 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.edwardraff.toyworld;

import jsat.linear.Vec;
import jsat.math.Function;

/**
 *
 * @author Edward Raff
 */
public interface Func1D extends Function
{

    @Override
    public default double f(Vec arg0)
    {
        return f(arg0.get(0));
    }

    @Override
    public default double f(double... arg0)
    {
        return f(arg0[0]);
    }
    
    /**
     * Function of one variable
     * @param x the input 
     * @return the output
     */
    public double f(double x);
}
