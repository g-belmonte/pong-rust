use std::any::Any;

use cgmath::{Matrix4, Vector3};

use engine::graphics_manager::structures::{hidden_transform, rect_mesh};
use engine::graphics_manager::{GraphicsManager, ModelHandle};
use engine::resources::{Mesh, Resources};
use engine::scene::Behaviour;

// Segment order: [top, top-left, top-right, middle, bottom-left, bottom-right, bottom].
#[rustfmt::skip]
const SEGMENTS_FOR_DIGIT: [[bool; 7]; 10] = [
    //   T      TL     TR     M      BL     BR     B
    [ true,  true,  true, false,  true,  true,  true], // 0
    [false, false,  true, false, false,  true, false], // 1
    [ true, false,  true,  true,  true, false,  true], // 2
    [ true, false,  true,  true, false,  true,  true], // 3
    [false,  true,  true,  true, false,  true, false], // 4
    [ true,  true, false,  true, false,  true,  true], // 5
    [ true,  true, false,  true,  true,  true,  true], // 6
    [ true, false,  true, false, false,  true, false], // 7
    [ true,  true,  true,  true,  true,  true,  true], // 8
    [ true,  true,  true,  true, false,  true,  true], // 9
];

const COLOR: [f32; 3] = [1.0, 1.0, 1.0];

/// Two shared meshes — horizontal and vertical segments — loaded once
/// and reused as instances by every digit. The RAII `Mesh` wrappers must
/// outlive every digit instance built against them.
pub struct DigitMeshes {
    pub horizontal: Mesh,
    pub vertical: Mesh,
    pub segment_size: f32,
}

impl DigitMeshes {
    pub fn load(
        resources: &mut Resources,
        gm: &mut GraphicsManager,
        segment_size: f32,
    ) -> Self {
        let thickness = segment_size * 0.2;
        Self {
            horizontal: resources.load_mesh(gm, &rect_mesh(segment_size, thickness)),
            vertical: resources.load_mesh(gm, &rect_mesh(thickness, segment_size)),
            segment_size,
        }
    }
}

/// Seven-segment digit. The `Object`'s `transform` positions the digit;
/// the seven segment instances are owned by the behaviour and contributed
/// each frame via `collect_renderables` with the parent matrix applied.
pub struct DigitBehaviour {
    segments: [ModelHandle; 7],
    segment_size: f32,
    value: u8,
}

impl DigitBehaviour {
    pub fn new(gm: &mut GraphicsManager, meshes: &DigitMeshes) -> Self {
        let h = meshes.horizontal.handle();
        let v = meshes.vertical.handle();
        let segments = [
            gm.register_instance(h, COLOR), // top
            gm.register_instance(v, COLOR), // top-left
            gm.register_instance(v, COLOR), // top-right
            gm.register_instance(h, COLOR), // middle
            gm.register_instance(v, COLOR), // bottom-left
            gm.register_instance(v, COLOR), // bottom-right
            gm.register_instance(h, COLOR), // bottom
        ];
        Self {
            segments,
            segment_size: meshes.segment_size,
            value: 0,
        }
    }

    pub fn set_value(&mut self, value: u8) {
        self.value = value;
    }
}

impl Behaviour for DigitBehaviour {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn collect_renderables(
        &self,
        parent_matrix: Matrix4<f32>,
        out: &mut Vec<(ModelHandle, Matrix4<f32>)>,
    ) {
        let lit = SEGMENTS_FOR_DIGIT[(self.value % 10) as usize];
        let s = self.segment_size;
        let half = s / 2.0;
        // Local segment offsets (positive Y is downwards).
        let offsets = [
            Vector3 { x: 0.0,    y: -s,    z: 0.0 }, // top
            Vector3 { x: -half,  y: -half, z: 0.0 }, // top-left
            Vector3 { x:  half,  y: -half, z: 0.0 }, // top-right
            Vector3 { x: 0.0,    y: 0.0,   z: 0.0 }, // middle
            Vector3 { x: -half,  y:  half, z: 0.0 }, // bottom-left
            Vector3 { x:  half,  y:  half, z: 0.0 }, // bottom-right
            Vector3 { x: 0.0,    y:  s,    z: 0.0 }, // bottom
        ];

        for i in 0..7 {
            let m = if lit[i] {
                parent_matrix * Matrix4::from_translation(offsets[i])
            } else {
                // Park unlit segments off-screen. Every registered instance
                // must emit a transform each frame or it would draw at the
                // previous frame's matrix.
                hidden_transform()
            };
            out.push((self.segments[i], m));
        }
    }

    fn on_despawn(&mut self, gm: &mut GraphicsManager) {
        for &h in &self.segments {
            gm.unregister_instance(h);
        }
    }
}
