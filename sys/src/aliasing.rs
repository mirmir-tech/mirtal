#[allow(let_underscore_drop, clippy::must_use_candidate, clippy::panic, clippy::too_many_arguments)]
#[cxx::bridge(namespace = "mirtal")]
pub mod ffi {
    unsafe extern "C++" {
        include!("mirtal/bridge.h");

        type Array = crate::ffi::Array;
        type Arrays = crate::ffi::Arrays;
        type AliasingPlan;
        type MetalLibrary = crate::ffi::MetalLibrary;
        type Stream = crate::ffi::Stream;

        fn new_aliasing_plan(
            input_arity: usize,
            constant_arity: usize,
            output_aliases: &[u32],
            stride_inputs: &[u32],
            stride_axes: &[u32],
            function: &str,
            library: &MetalLibrary,
        ) -> Result<UniquePtr<AliasingPlan>>;

        fn aliasing_plan_set_input(
            plan: Pin<&mut AliasingPlan>,
            index: usize,
            input: &Array,
        ) -> Result<()>;

        fn aliasing_plan_dispatch(
            plan: Pin<&mut AliasingPlan>,
            constants: &[u32],
            grid_x: u32,
            grid_y: u32,
            grid_z: u32,
            group_x: u32,
            group_y: u32,
            group_z: u32,
            stream: &Stream,
        ) -> Result<UniquePtr<Arrays>>;

        fn aliasing_dispatch(
            inputs: &Arrays,
            output_aliases: &[u32],
            constants: &[u32],
            stride_inputs: &[u32],
            stride_axes: &[u32],
            function: &str,
            library: &MetalLibrary,
            grid_x: u32,
            grid_y: u32,
            grid_z: u32,
            group_x: u32,
            group_y: u32,
            group_z: u32,
            stream: &Stream,
        ) -> Result<UniquePtr<Arrays>>;
    }
}
