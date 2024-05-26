import { terser } from "rollup-plugin-terser";
import { nodeResolve } from "@rollup/plugin-node-resolve";
import babel from "@rollup/plugin-babel";
import pkg from './package.json';

const generate_config_umd = ({ minify = false }) => ({
     input: 'src/index.js',
     plugins: [
         nodeResolve(),
         babel({
             babelHelpers: 'bundled',
             exclude: 'node_modules/**'
         }),
         ...(minify ? [terser()] : [])
     ],
     output: {
         format: 'umd',
         file: minify ? pkg.browser.replace('.js', '.min.js') : pkg.browser,
         name: 'RadixSort',
         esModule: false,
         exports: 'named',
         sourcemap: true,
     }
})

const generate_config_esm_cjs = () => ({
    input: 'src/index.js',
    plugins: [
        nodeResolve(),
    ],
    output: [
        {
            format: 'es',
            file: pkg.module,
            exports: 'named',
            sourcemap: true,
        },
        {
            format: 'cjs',
            file: pkg.main,
            exports: 'named',
            sourcemap: true,
        }
    ]
})

export default [
    // UMD
    generate_config_umd({ minify: false }),
    // UMD (minified)
    generate_config_umd({ minify: true }),
    // ESM and CJS
    generate_config_esm_cjs()
]