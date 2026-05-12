use cgmath::{Matrix4, Vector3};

use engine::graphics_manager::structures::{hidden_transform, rect_mesh};
use engine::graphics_manager::{GraphicsManager, ModelHandle};
use engine::resources::{Mesh, Resources};

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
/// and reused as instances by every digit. Saves 12 buffer allocations
/// (relative to the old per-segment registration) for two on-screen digits.
/// The RAII `Mesh` wrappers must outlive every digit instance built against them.
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

pub struct Digit {
    pub segments: [ModelHandle; 7],
    pub position: Vector3<f32>,
    pub segment_size: f32,
    pub value: u8,
}

impl Digit {
    pub fn new(gm: &mut GraphicsManager, meshes: &DigitMeshes, position: Vector3<f32>) -> Self {
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
            position,
            segment_size: meshes.segment_size,
            value: 0,
        }
    }

    pub fn set_value(&mut self, value: u8) {
        self.value = value;
    }

    pub fn get_model_transforms(&self) -> Vec<(ModelHandle, Matrix4<f32>)> {
        let lit = SEGMENTS_FOR_DIGIT[(self.value % 10) as usize];
        let s = self.segment_size;
        let half = s / 2.0;
        // Positive Y is downwards.
        let offsets = [
            Vector3 { x: 0.0,    y: -s,   z: 0.0 }, // top
            Vector3 { x: -half,  y: -half, z: 0.0 }, // top-left
            Vector3 { x:  half,  y: -half, z: 0.0 }, // top-right
            Vector3 { x: 0.0,    y: 0.0,  z: 0.0 }, // middle
            Vector3 { x: -half,  y:  half, z: 0.0 }, // bottom-left
            Vector3 { x:  half,  y:  half, z: 0.0 }, // bottom-right
            Vector3 { x: 0.0,    y:  s,   z: 0.0 }, // bottom
        ];

        // Unlit segments are parked off-screen rather than skipped: every
        // registered instance must produce a transform each frame, otherwise
        // it would render at its previous position.
        (0..7)
            .map(|i| {
                let m = if lit[i] {
                    Matrix4::from_translation(self.position + offsets[i])
                } else {
                    hidden_transform()
                };
                (self.segments[i], m)
            })
            .collect()
    }
}

