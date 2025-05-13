from openfermion.ops import QubitOperator

h_lih = (
        -7.4989469 * QubitOperator('') +
        -0.0029329 * QubitOperator('Y0 Y1 X2 X3') +
        0.0029329 * QubitOperator('X0 Y1 Y2 X3') +
        0.0129108 * QubitOperator('X1 Z2 X3') +
        -0.0013743 * QubitOperator('Z0 X1 Z2 X3') +
        0.0115364 * QubitOperator('X1 X3') +
        0.0029329 * QubitOperator('Y0 X1 X2 Y3') +
        -0.0029320 * QubitOperator('X0 X1 Y2 Y3') +
        0.0129108 * QubitOperator('Y1 Z2 Y3') +
        -0.0013743 * QubitOperator('Z0 Y1 Z2 Y3') +
        0.0115364 * QubitOperator('X0 Z1 X2 Z3') +
        0.1619948 * QubitOperator('Z3') +
        0.0115364 * QubitOperator('Z2 Z3') +
        0.0115364 * QubitOperator('Y0 Z1 Y2 Z3') +
        0.1244477 * QubitOperator('Z2 Z3') + # 14
        0.0541304 * QubitOperator('Z1 Z3') +
        0.0570634 * QubitOperator('Z0 Z3') +
        0.0129108 * QubitOperator('X0 Z1 X2') +
        -0.0013743 * QubitOperator('X0 X2') +
        0.0129107 * QubitOperator('Y0 Z1 Y2') +
        -0.0013743 * QubitOperator('Y0 Y2') +
        0.1619948 * QubitOperator('Z2') +
        0.0570634 * QubitOperator('Z1 Z2') +
        0.0541304 * QubitOperator('Z0 Z2') +
        -0.0132437 * QubitOperator('Z1') +
        0.0847961 * QubitOperator('Z0 Z1') +
        -0.0132436 * QubitOperator('Z0')
    )