void getEigenValuesVectors(in float mat_data[3][3], out float vectors[3][3], out float values[3])
{
    vec3 e = vec3(0);

    int dimension = 3;
    int dimensionMinusOne = 2;

    for (int j = 0; j < dimension; ++j)
        values[j] = mat_data[dimensionMinusOne][j];

    // Householder reduction to tridiagonal form.
    for (int i = dimensionMinusOne; i > 0 && i <= dimensionMinusOne; --i)
    {
        // Scale to avoid under/overflow.
        float scale = 0.0;
        float h = 0.0;
        for (int k = 0; k < i; ++k)
        {
            scale += abs(values[k]);
        }

        if (scale == 0.0)
        {
            e[i] = values[i - 1];

            for (int j = 0; j < i; ++j)
            {
                values[j] = mat_data[(i - 1)][j];
                mat_data[i][j] = 0.0;
                mat_data[j][i] = 0.0;
            }
        }
        else
        {
            // Generate Householder vector.
            for (int k = 0; k < i; ++k)
            {
                values[k] /= scale;
                h += values[k] * values[k];
            }

            float f = values[i - 1];
            float g = sqrt(h);

            if (f > 0.0)
            {
                g = -g;
            }

            e[i] = scale * g;
            h -= f * g;
            values[i - 1] = f - g;

            for (int j = 0; j < i; ++j)
            {
                e[j] = 0.0;
            }

            // Apply similarity transformation to remaining columns.
            for (int j = 0; j < i; ++j)
            {
                f = values[j];
                mat_data[j][i] = f;
                g = e[j] + mat_data[j][j] * f;

                for (int k = j + 1; k <= i - 1; ++k)
                {
                    g += mat_data[k][j] * values[k];
                    e[k] += mat_data[k][j] * f;
                }

                e[j] = g;
            }

            f = 0.0;
            for (int j = 0; j < i; ++j)
            {
                e[j] /= h;
                f += e[j] * values[j];
            }

            float hh = f / (h + h);

            for (int j = 0; j < i; ++j)
            {
                e[j] -= hh * values[j];
            }

            for (int j = 0; j < i; ++j)
            {
                f = values[j];
                g = e[j];

                for (int k = j; k <= i - 1; ++k)
                {
                    mat_data[k][j] = mat_data[k][j] - (f * e[k] + g * values[k]);
                }

                values[j] = mat_data[i - 1][j];
                mat_data[i][j] = 0.0;
            }
        }
        values[i] = h;
    }

    // Accumulate transformations.
    for (int i = 0; i < dimensionMinusOne; ++i)
    {
        mat_data[dimensionMinusOne][i] = mat_data[i][i];
        mat_data[i][i] = 1.0;
        float h = values[i + 1];

        if (h != 0.0)
        {
            for (int k = 0; k <= i; ++k)
            {
                values[k] = mat_data[k][i + 1] / h;
            }

            for (int j = 0; j <= i; ++j)
            {
                float g = 0.0;

                for (int k = 0; k <= i; ++k)
                {
                    g += mat_data[k][i + 1] * mat_data[k][j];
                }

                for (int k = 0; k <= i; ++k)
                {
                    mat_data[k][j] = mat_data[k][j] - (g * values[k]);
                }
            }
        }
        for (int k = 0; k <= i; ++k)
        {
            mat_data[k][i + 1] = 0.0;
        }
    }

    for (int j = 0; j < dimension; ++j)
    {
        values[j] = mat_data[dimensionMinusOne][j];
        mat_data[dimensionMinusOne][j] = 0.0;
    }

    mat_data[dimensionMinusOne][dimensionMinusOne] = 1.0;
    e[0] = 0.0;

    for (int i = 1; i < dimension; ++i)
        e[i - 1] = e[i];

    e[dimensionMinusOne] = 0.0;

    float f = float(0.0);
    float tst1 = float(0.0);
    float eps = float(pow(2.0, -52.0));
    for (int l = 0; l < dimension; ++l)
    {
        // Find small subdiagonal element
        tst1 = float(max(tst1, abs(values[l]) + abs(e[l])));
        int m = l;
        while (m < dimension)
        {
            if (abs(e[m]) <= eps * tst1) break;
            ++m;
        }

        // If m == l, d[l] is an eigenvalue,
        // otherwise, iterate.
        if (m > l && l < 2)
        {
            int iter = 0;
            do
            {
                ++iter;  // (Could check iteration count here.)
                // Compute implicit shift
                float g = values[l];
                float p = (values[l + 1] - g) / (float(2.0) * e[l]);
                float r = float(sqrt(p * p + float(1.0) * float(1.0)));
                if (p < 0) r = -r;
                values[l] = e[l] / (p + r);
                values[l + 1] = e[l] * (p + r);
                float dl1 = values[l + 1];
                float h = g - values[l];
                for (int i = l + 2; i < dimension; ++i)
                    values[i] -= h;
                f = f + h;

                // Implicit QL transformation.
                p = values[m];
                float c = float(1.0);
                float c2 = c;
                float c3 = c;
                float el1 = e[l + 1];
                float s = float(0.0);
                float s2 = float(0.0);
                for (int i = m - 1; i >= l && i <= m - 1; --i)
                {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = float(sqrt(p * p + e[i] * e[i]));
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * values[i] - s * g;
                    values[i + 1] = h + s * (c * g + s * values[i]);

                    // Accumulate transformation.
                    for (int k = 0; k < dimension; ++k)
                    {
                        h = mat_data[k][i + 1];
                        mat_data[k][i + 1] = (s * mat_data[k][i] + c * h);
                        mat_data[k][i] = (c * mat_data[k][i] - s * h);
                    }
                }

                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                values[l] = c * p;
                // Check for convergence.
            } while (abs(e[l]) > eps * tst1 && iter < 30);
        }
        values[l] = values[l] + f;
        e[l] = float(0.0);
    }

    // Sort eigenvalues and corresponding vectors.
    for (int i = 0; i < dimensionMinusOne; ++i)
    {
        int k = i;
        float p = values[i];

        for (int j = i + 1; j < dimension; ++j)
        {
            if (values[j] < p)
            {
                k = j;
                p = values[j];
            }
        }
        if (k != i)
        {
            values[k] = values[i];
            values[i] = p;
            for (int j = 0; j < dimension; ++j)
            {
                p = mat_data[j][i];
                mat_data[j][i] = mat_data[j][k];
                mat_data[j][k] = p;
            }
        }
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            vectors[i][j] = mat_data[i][j];
}